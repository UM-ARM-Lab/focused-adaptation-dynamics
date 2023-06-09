import pathlib
from typing import Dict, Optional

import numpy as np
from colorama import Fore
from tqdm import tqdm
import rospy
from analysis import results_utils
from analysis.results_utils import NoTransitionsError
from arc_utilities.algorithms import reversed_chunked
from link_bot_data.dataset_constants import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT
from link_bot_data.split_dataset import split_dataset
from link_bot_data.tf_dataset_utils import write_example
from link_bot_planning.my_planner import PlanningResult
from link_bot_planning.trial_result import ExecutionResult
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_hjson
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_np_arrays
from moonshine.numpify import numpify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def compute_example_idx(trial_idx, example_idx_for_trial):
    return 10_000 * trial_idx + example_idx_for_trial


class ResultsToDynamicsDataset:

    def __init__(self,
                 results_dir: pathlib.Path,
                 outname: str,
                 visualize: bool,
                 traj_length: Optional[int] = None,
                 val_split=DEFAULT_VAL_SPLIT,
                 test_split=DEFAULT_TEST_SPLIT,
                 root=pathlib.Path("fwd_model_data")):
        self.visualize = visualize
        self.traj_length = traj_length
        self.results_dir = results_dir
        self.trials = (list(results_utils.trials_generator(self.results_dir)))
        self.outdir = root / outname
        self.val_split = val_split
        self.test_split = test_split

        self.scenario, self.metadata = results_utils.get_scenario_and_metadata(results_dir)

        self.example_idx = None

        self.outdir.mkdir(exist_ok=True, parents=True)

    def run(self, data_collection_params_fn=None):
        self.save_hparams(data_collection_params_fn=data_collection_params_fn)
        self.results_to_dynamics_dataset()
        split_dataset(self.outdir, val_split=self.val_split, test_split=self.test_split)

        return self.outdir

    def save_hparams(self, data_collection_params_fn=None):
        # FIXME: hard-coded
        planner_params = self.metadata['planner_params']
        dataset_hparams = {
            'scenario': planner_params['scenario'],
            'from_results': self.results_dir,
            'seed': None,
            'n_trajs': len(self.trials)
        }
        if data_collection_params_fn is None:
            data_collection_params = {
                                         'scenario_params': planner_params.get("scenario_params", {}),
                                         'max_step_size': planner_params.get("max_step_size", 0.01),
                                         'max_distance_gripper_can_move': 0.1,
                                         'res': 0.02,
                                         'service_provider': 'gazebo',
                                         'state_description': {
                                             'left_gripper': 3,
                                             'right_gripper': 3,
                                             'joint_positions': 18,
                                             'rope': 75,
                                         },
                                         'state_metadata_description': {
                                             'joint_names': None,
                                         },
                                         'action_description': {
                                             'left_gripper_position': 3,
                                             'right_gripper_position': 3,
                                         },
                                         'env_description': {
                                             'env': None,
                                             'extent': 4,
                                             'origin_point': 3,
                                             'res': None,
                                             'scene_msg': None,
                                         },
                                     }
        else:
            data_collection_params = load_hjson(data_collection_params_fn)
        dataset_hparams["data_collection_params"] = data_collection_params
        with (self.outdir / 'hparams.hjson').open('w') as dataset_hparams_file:
            my_hdump(dataset_hparams, dataset_hparams_file, indent=2)

    def results_to_dynamics_dataset(self):
        if self.visualize:
            dataset = TorchDynamicsDataset(self.outdir, mode='', is_empty=True)

        total_examples = 0
        for trial_idx, datum, _ in self.trials:
            self.scenario.heartbeat()

            example_idx_for_trial = 0

            self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
            try:
                examples_gen = self.result_datum_to_dynamics_dataset(datum, trial_idx)
                for example in tqdm(examples_gen):
                    self.example_idx = compute_example_idx(trial_idx, example_idx_for_trial)
                    total_examples += 1
                    write_example(self.outdir, example, self.example_idx, 'pkl')
                    if self.visualize:
                        dataset.anim_rviz(example)
                    example_idx_for_trial += 1
            except NoTransitionsError:
                rospy.logerr(f"Trial {trial_idx} had no transitions")
                pass

        print(Fore.LIGHTMAGENTA_EX + f"Wrote {total_examples} examples" + Fore.RESET)

    def result_datum_to_dynamics_dataset(self, datum: Dict, trial_idx: int):
        steps = datum['steps']

        if len(steps) == 0:
            raise NoTransitionsError()

        actions = []
        states = []
        states_step = None
        actions_step = None
        for step_idx, step in enumerate(steps):
            if step['type'] == 'executed_plan':
                planning_result: PlanningResult = step['planning_result']
                execution_result: ExecutionResult = step['execution_result']
                actions_step = planning_result.actions
                states_step = execution_result.path
            elif step['type'] == 'executed_recovery':
                execution_result: ExecutionResult = step['execution_result']
                recovery_action = step['recovery_action']
                actions_step = [recovery_action]
                states_step = execution_result.path
            else:
                raise NotImplementedError(f"invalid step type {step['type']}")

            if len(actions_step) == 0 or actions_step[0] is None:
                continue

            if len(actions_step) >= len(states_step):
                # indicates stop on error occurred
                num_states = len(states_step)
                actions_step = actions_step[:num_states - 1]

            actions_step = numpify(actions_step)
            # NOTE: here we append the final action to make action & state the same length
            actions_step.append(actions_step[-1])
            states_step = numpify(states_step)

            actions.extend(actions_step)
            states.extend(states_step)

        if self.traj_length is not None:
            time_mask = np.ones(self.traj_length, np.float32)
            if len(states) < self.traj_length:
                time_mask[len(states):] = 0
                n_pad = self.traj_length - len(states)
                pad_state = {}
                for k, v in states_step[-1].items():
                    if k == 'joint_names':
                        pad_state[k] = v
                    else:
                        pad_state[k] = np.zeros_like(v)
                pad_action = {k: np.zeros_like(v) for k, v in actions_step[-1].items()}
                states_padded = states + n_pad * [pad_state]
                actions_padded = actions + (n_pad - 1) * [pad_action]
                state_subsequences = [states_padded]
                action_subsequences = [actions_padded]
            else:
                state_subsequences = reversed_chunked(states, self.traj_length)
                action_subsequences = reversed_chunked(actions, self.traj_length)
                action_subsequences = [aseq[:-1] for aseq in action_subsequences]
        else:
            time_mask = np.ones_like(states).astype(np.float32)
            action_subsequences = [actions]
            state_subsequences = [states]

        for action_subsequence, state_subsequence in zip(action_subsequences, state_subsequences):
            actions_dict = sequence_of_dicts_to_dict_of_np_arrays(action_subsequence, 0)
            states_dict = sequence_of_dicts_to_dict_of_np_arrays(state_subsequence, 0)

            time_idx = np.arange(len(state_subsequence), dtype=np.float32)
            environment = steps[0]['planning_query'].environment
            example = {
                'traj_idx': trial_idx,
                'time_idx': time_idx,
            }
            example.update(environment)
            example.update(actions_dict)
            example.update(states_dict)

            example.pop("stdev", None)
            example.pop("error", None)
            example.pop("num_diverged", None)
            example.pop("left_gripper_delta_position", None)
            example.pop("right_gripper_delta_position", None)
            example['time_mask'] = time_mask

            yield example
