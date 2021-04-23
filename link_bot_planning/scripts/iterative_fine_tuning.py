#!/usr/bin/env python
import argparse
import itertools
import logging
import pathlib
import pickle
import random
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

import numpy as np
from more_itertools import chunked
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, Metric

from arc_utilities.algorithms import nested_dict_update
from link_bot_classifiers.fine_tune_classifier import fine_tune_classifier
from link_bot_classifiers.nn_classifier import NNClassifier
from link_bot_classifiers.points_collision_checker import PointsCollisionChecker
from link_bot_data.files_dataset import FilesDataset
from link_bot_gazebo import gazebo_services
from link_bot_gazebo.gazebo_services import get_gazebo_processes
from link_bot_planning.analysis.results_metrics import load_analysis_params, generate_per_trial_metrics, Successes
from link_bot_planning.get_planner import get_planner, load_classifier
from link_bot_planning.results_to_classifier_dataset import ResultsToClassifierDataset
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.heartbeat import HeartBeat
from link_bot_pycommon.pycommon import pathify, deal_with_exceptions
from moonshine.metrics import LossMetric
from moonshine.moonshine_utils import repeat, add_batch
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors as tt
from moonshine.my_keras_model import MyKerasModel

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou

import colorama
import hjson
import tensorflow as tf
from colorama import Fore, Style

import rospy
from arc_utilities import ros_init
from link_bot_data.dataset_utils import data_directory, compute_batch_size, tf_write_example, add_predicted, \
    replaced_true_with_predicted
from link_bot_planning.planning_evaluation import load_planner_params, EvaluatePlanning
from link_bot_pycommon.args import run_subparsers
from link_bot_pycommon.job_chunking import JobChunker
from moonshine.filepath_tools import load_hjson, load_params


@dataclass
class IterationData:
    iteration: int
    iteration_chunker: JobChunker
    fine_tuning_dataset_dirs: List[pathlib.Path]
    latest_classifier_checkpoint_dir: pathlib.Path
    latest_recovery_checkpoint_dir: pathlib.Path


class IterativeFineTuning:

    def __init__(self,
                 log: Dict,
                 no_execution: bool,
                 timeout: int,
                 on_exception: str,
                 logfile_name: pathlib.Path,
                 ):
        self.no_execution = no_execution
        self.on_exception = on_exception
        self.log = log
        self.ift_config = self.log['ift_config']
        self.initial_planner_params = pathify(self.log['planner_params'])
        self.log_full_tree = False
        self.initial_planner_params["log_full_tree"] = self.log_full_tree
        self.initial_planner_params['classifier_model_dir'] = []  # this gets replace at every iteration
        self.test_scenes_dir = pathlib.Path(self.log['test_scenes_dir'])
        self.verbose = -1
        self.labeling_params = load_hjson(pathlib.Path('labeling_params/classifier/dual.hjson'))
        self.labeling_params = nested_dict_update(self.labeling_params, self.ift_config.get('labeling_params_update', {}))
        self.initial_planner_params = nested_dict_update(self.initial_planner_params, self.ift_config.get('planner_params_update', {}))
        self.collision_pretraining_config = self.ift_config.get('collision_pretraining', {})

        # DEBUGGING
        if timeout is not None:
            rospy.loginfo(f"Overriding with timeout {timeout}")
            self.initial_planner_params["termination_criteria"]['timeout'] = timeout
            self.initial_planner_params["termination_criteria"]['total_timeout'] = timeout
        # DEBUGGING

        self.gazebo_processes = get_gazebo_processes()

        self.outdir = logfile_name.parent

        self.job_chunker = JobChunker(logfile_name=logfile_name)
        self.trials_directory = self.outdir / 'classifier_training_logdir'
        self.planning_results_root_dir = self.outdir / 'planning_results'

        all_trial_indices = list(get_all_scene_indices(self.test_scenes_dir))
        trials_generator_type = self.ift_config['trials_generator_type']
        if trials_generator_type == 'cycle':
            self.trial_indices_generator = chunked(itertools.cycle(all_trial_indices),
                                                   self.ift_config['trials_per_iteration'])
        elif trials_generator_type == 'random':
            def _random():
                rng = random.Random(0)
                while True:
                    yield rng.choice(all_trial_indices)

            self.trial_indices_generator = chunked(_random(),
                                                   self.ift_config['trials_per_iteration'])
        else:
            raise NotImplementedError(f"Unimplemented {trials_generator_type}")

        # Start Services
        [p.resume() for p in self.gazebo_processes]
        self.service_provider = gazebo_services.GazeboServices()
        self.service_provider.play()  # time needs to be advancing while we setup the planner

        # Setup scenario
        self.scenario = get_scenario(self.initial_planner_params["scenario"])
        self.scenario.on_before_get_state_or_execute_action()
        self.service_provider.setup_env(verbose=self.verbose,
                                        real_time_rate=self.initial_planner_params['real_time_rate'],
                                        max_step_size=0.01,  # FIXME:
                                        play=True)

        self.planner = get_planner(planner_params=self.initial_planner_params,
                                   verbose=self.verbose,
                                   log_full_tree=self.log_full_tree,
                                   scenario=self.scenario)

        # Emits a message every 10 seconds,
        # so that another node can monitor for the program getting hung-up and restart it
        self.heartbeat = HeartBeat(10)

    def run(self, num_fine_tuning_iterations: int):
        initial_classifier_checkpoint = pathlib.Path(self.log['initial_classifier_checkpoint'])
        initial_recovery_checkpoint = pathlib.Path(self.log['initial_recovery_checkpoint'])
        fine_tuning_dataset_dirs = []

        # Pre-adaptation training steps
        if self.collision_pretraining_config.get('use_collision_pretraining', False):
            print("Collision Pretraining")
            [p.suspend() for p in self.gazebo_processes]
            self.collision_pretraining(initial_classifier_checkpoint)

        latest_classifier_checkpoint_dir = initial_classifier_checkpoint
        latest_recovery_checkpoint_dir = initial_recovery_checkpoint
        for iteration_idx in range(num_fine_tuning_iterations):
            jobkey = f"iteration {iteration_idx}"
            iteration_chunker = self.job_chunker.sub_chunker(jobkey)
            iteration_start_time = iteration_chunker.get_result('start_time')
            if iteration_start_time is None:
                iteration_start_time = perf_counter()
                iteration_chunker.store_result('start_time', iteration_start_time)
            iteration_data = IterationData(fine_tuning_dataset_dirs=fine_tuning_dataset_dirs,
                                           iteration=iteration_idx,
                                           iteration_chunker=iteration_chunker,
                                           latest_classifier_checkpoint_dir=latest_classifier_checkpoint_dir,
                                           latest_recovery_checkpoint_dir=latest_recovery_checkpoint_dir,
                                           )
            # planning
            planning_results_dir = self.plan_and_execute(iteration_data)

            # DEBUGGING
            # # convert results to classifier dataset
            # new_dataset_dir = self.update_datasets(iteration_data, planning_results_dir)
            # iteration_data.fine_tuning_dataset_dirs.append(new_dataset_dir)
            # # fine tune (on all of the classifier datasets so far)
            #
            # latest_classifier_checkpoint_dir = self.fine_tune(iteration_data)
            # DEBUGGING

            # TODO: add fine tuning recovery
            iteration_end_time = iteration_chunker.get_result('end_time')
            if iteration_end_time is None:
                iteration_end_time = perf_counter()
                iteration_chunker.store_result('end_time', iteration_end_time)
            iteration_time = iteration_end_time - iteration_start_time
            print(Style.BRIGHT + f"Finished iteration {iteration_idx}, {iteration_time:.1f}s" + Style.RESET_ALL)

        [p.kill() for p in self.gazebo_processes]

    def collision_pretraining(self, initial_classifier_checkpoint: pathlib.Path):
        # create a dataset (unlabeled)
        dataset_dir = self.generate_collision_pretraining_dataset(initial_classifier_checkpoint)

        # fine-tune the classifier
        def compute_loss(self: NNClassifier, dataset_element, outputs):
            labels = tf.expand_dims(dataset_element['not_in_collision'], axis=2)
            logits = outputs['logits']
            bce = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits, from_logits=True)

            return {
                'loss': bce,
            }

        def create_metrics(self: NNClassifier):
            MyKerasModel.create_metrics(self)
            return {
                'accuracy':  BinaryAccuracy(),
                'precision': Precision(),
                'recall':    Recall(),
                'loss':      LossMetric(),
            }

        def compute_metrics(self: NNClassifier, metrics: Dict[str, Metric], losses: Dict, dataset_element, outputs):
            labels = tf.expand_dims(dataset_element['not_in_collision'], axis=2)
            probabilities = outputs['probabilities']
            metrics['accuracy'].update_state(y_true=labels, y_pred=probabilities)
            metrics['precision'].update_state(y_true=labels, y_pred=probabilities)
            metrics['recall'].update_state(y_true=labels, y_pred=probabilities)

        new_latest_checkpoint_dir = fine_tune_classifier(dataset_dirs=[dataset_dir],
                                                         checkpoint=initial_classifier_checkpoint,
                                                         log=f'collision_pretraining_logdir',
                                                         trials_directory=self.outdir,
                                                         batch_size=32,
                                                         verbose=self.verbose,
                                                         validate_first=True,
                                                         compute_loss=compute_loss,
                                                         create_metrics=create_metrics,
                                                         compute_metrics=compute_metrics,
                                                         **self.collision_pretraining_config)
        return new_latest_checkpoint_dir

    def generate_collision_pretraining_dataset(self, initial_classifier_checkpoint: pathlib.Path):
        # FIXME:
        # I should actually the run planner and use ResultsToClassifier dataset with the full-tree option?

        dataset_dir = self.outdir / 'pretraining_dataset'
        dataset_dir.mkdir(exist_ok=True)

        cc = PointsCollisionChecker(pathlib.Path('cl_trials/cc_baseline/none'), self.scenario)

        # write hparams file
        new_hparams_filename = dataset_dir / 'hparams.hjson'
        classifier_hparams = load_params(initial_classifier_checkpoint)
        new_dataset_hparams = classifier_hparams["classifier_dataset_hparams"]
        with new_hparams_filename.open('w') as new_hparams_file:
            hjson.dump(new_dataset_hparams, new_hparams_file)

        files_dataset = FilesDataset(root_dir=dataset_dir)

        def configs_generator():
            rng = random.Random(0)
            configs_dir = pathlib.Path(self.collision_pretraining_config['configs_dir'])
            configs_repeated = list(configs_dir.glob("initial_config_*.pkl"))
            while True:
                config_filename = rng.choice(configs_repeated)
                config = pickle.load(config_filename.open("rb"))
                yield config['state'], config['env']

        n_examples = self.collision_pretraining_config['n_examples']
        action_rng = np.random.RandomState(0)
        batch_size = 16
        action_sequence_length = 10
        action_params_filename = pathlib.Path(self.collision_pretraining_config['action_params_filename'])
        action_params = load_hjson(action_params_filename)
        for example_idx in range(n_examples):
            # get environment, this is the magic where we use the fact that we have the environment description
            state, environment = next(configs_generator())

            # sample action sequences
            state_batched = repeat(state, batch_size, 0, True)
            environment_batched = {k: [v] * batch_size for k, v in environment.items()}
            actions_list = self.scenario.sample_action_sequences(environment=environment,
                                                                 state=state_batched,
                                                                 action_params=action_params,
                                                                 n_action_sequences=batch_size,
                                                                 action_sequence_length=action_sequence_length,
                                                                 validate=True,
                                                                 action_rng=action_rng)
            actions = tt([tt(a) for a in actions_list])
            state_batch_time = add_batch(state_batched, batch_axis=1)

            # pass through forward model
            states, _ = self.planner.fwd_model.propagate_tf_batched(environment=environment_batched,
                                                                    state=state_batch_time,
                                                                    actions=actions)

            # make example
            example = {
                'classifier_start_t': 0,
                'classifier_end_t':   2,
                'prediction_start_t': 0,
                'traj_idx':           example_idx,
                'time_idx':           [0, 1],
            }
            example.update(environment_batched)
            example.update({add_predicted(k): v for k, v in states.items()})
            example.update(actions)

            # check collision
            example_for_cc = replaced_true_with_predicted(example)
            not_in_collision = cc.label_in_collision(example_for_cc,
                                                     batch_size=batch_size,
                                                     state_sequence_length=(action_sequence_length + 1))
            example['not_in_collision'] = not_in_collision

            # split the sequence into individual transitions

            # write example
            full_filename = tf_write_example(dataset_dir, example, example_idx)
            files_dataset.add(full_filename)

        files_dataset.split()

        return dataset_dir

    def plan_and_execute(self, iteration_data: IterationData):
        i = iteration_data.iteration
        trials = next(self.trial_indices_generator)
        planning_chunker = iteration_data.iteration_chunker.sub_chunker('planning')
        planning_results_dir = pathify(planning_chunker.get_result('planning_results_dir'))
        if planning_results_dir is None:
            planning_results_dir = self.planning_results_root_dir / f'iteration_{i:04d}_planning'
            latest_classifier_checkpoint = iteration_data.latest_classifier_checkpoint_dir / 'best_checkpoint'
            latest_recovery_checkpoint = iteration_data.latest_recovery_checkpoint_dir / 'best_checkpoint'
            planner_params = self.initial_planner_params.copy()
            planner_params['recovery']['recovery_model_dir'] = latest_recovery_checkpoint
            planner_params['classifier_model_dir'] = [
                latest_classifier_checkpoint,
                pathlib.Path('cl_trials/new_feasibility_baseline/none'),
            ]
            self.initial_planner_params['fine_tuning_iteration'] = i

            # planning evaluation
            [p.resume() for p in self.gazebo_processes]
            classifier_models = load_classifier(planner_params, self.scenario)
            self.planner.classifier_models = classifier_models

            runner = EvaluatePlanning(planner=self.planner,
                                      service_provider=self.service_provider,
                                      job_chunker=planning_chunker,
                                      verbose=self.verbose,
                                      planner_params=planner_params,
                                      outdir=planning_results_dir,
                                      trials=trials,
                                      test_scenes_dir=self.test_scenes_dir)

            deal_with_exceptions(how_to_handle=self.on_exception, function=runner.run)
            [p.suspend() for p in self.gazebo_processes]

            analysis_params = load_analysis_params()
            metrics = generate_per_trial_metrics(analysis_params=analysis_params,
                                                 subfolders_ordered=[planning_results_dir],
                                                 method_names=[self.initial_planner_params['method_name']])
            successes = metrics[Successes].values[self.initial_planner_params['method_name']]
            latest_success_rate = successes.sum() / successes.shape[0]
            planning_chunker.store_result('latest_success_rate', latest_success_rate)
        else:
            latest_success_rate = planning_chunker.get_result('latest_success_rate')

        print(Fore.CYAN + f"Iteration {i} {latest_success_rate * 100:.1f}%")
        return planning_results_dir

    def update_datasets(self, iteration_data: IterationData, planning_results_dir):
        i = iteration_data.iteration
        dataset_chunker = iteration_data.iteration_chunker.sub_chunker('dataset')
        new_dataset_dir = pathify(dataset_chunker.get_result('new_dataset_dir'))
        if new_dataset_dir is None:
            [p.suspend() for p in self.gazebo_processes]

            new_dataset_dir = self.outdir / 'classifier_datasets' / f'iteration_{i:04d}_dataset'
            r = ResultsToClassifierDataset(results_dir=planning_results_dir,
                                           outdir=new_dataset_dir,
                                           labeling_params=self.labeling_params,
                                           verbose=self.verbose,
                                           **self.ift_config['results_to_dataset'])
            r.run()
            dataset_chunker.store_result('new_dataset_dir', new_dataset_dir.as_posix())
        return new_dataset_dir

    def fine_tune(self, iteration_data: IterationData):
        i = iteration_data.iteration
        latest_checkpoint = iteration_data.latest_classifier_checkpoint_dir / 'best_checkpoint'
        fine_tune_chunker = iteration_data.iteration_chunker.sub_chunker('fine tune')
        new_latest_checkpoint_dir = pathify(fine_tune_chunker.get_result('new_latest_checkpoint_dir'))
        if new_latest_checkpoint_dir is None:
            [p.suspend() for p in self.gazebo_processes]

            adaptive_batch_size = compute_batch_size(iteration_data.fine_tuning_dataset_dirs, max_batch_size=16)
            new_latest_checkpoint_dir = fine_tune_classifier(dataset_dirs=iteration_data.fine_tuning_dataset_dirs,
                                                             checkpoint=latest_checkpoint,
                                                             log=f'iteration_{i:04d}_training_logdir',
                                                             trials_directory=self.trials_directory,
                                                             batch_size=adaptive_batch_size,
                                                             verbose=self.verbose,
                                                             validate_first=True,
                                                             **self.ift_config)
            fine_tune_chunker.store_result('new_latest_checkpoint_dir', new_latest_checkpoint_dir.as_posix())
        return new_latest_checkpoint_dir


def start_iterative_fine_tuning(nickname: str,
                                planner_params_filename: pathlib.Path,
                                classifier_checkpoint: pathlib.Path,
                                recovery_checkpoint: pathlib.Path,
                                ift_config_filename: pathlib.Path,
                                num_fine_tuning_iterations: int,
                                no_execution: bool,
                                timeout: int,
                                test_scenes_dir: pathlib.Path,
                                on_exception: str,
                                ):
    # DEBUGGING
    # from_env = input("from: ")
    # to_env = input("to: ")
    from_env = "debugging"
    to_env = "debugging"
    # DEBUGGING

    # setup
    outdir = data_directory(pathlib.Path('results') / 'iterative_fine_tuning' / f"{nickname}")

    if not outdir.exists():
        rospy.loginfo(Fore.YELLOW + "Creating output directory: {}".format(outdir))
        outdir.mkdir(parents=True)

    ift_config = load_hjson(ift_config_filename)

    initial_planner_params = load_planner_params(planner_params_filename)

    logfile_name = outdir / 'logfile.hjson'
    log = {
        'nickname':                      nickname,
        'planner_params':                initial_planner_params,
        'test_scenes_dir':               test_scenes_dir.as_posix(),
        'initial_classifier_checkpoint': classifier_checkpoint.as_posix(),
        'initial_recovery_checkpoint':   recovery_checkpoint.as_posix(),
        'from_env':                      from_env,
        'to_env':                        to_env,
        'ift_config':                    ift_config,
    }
    with logfile_name.open("w") as logfile:
        hjson.dump(log, logfile)

    ift = IterativeFineTuning(log=log,
                              no_execution=no_execution,
                              timeout=timeout,
                              on_exception=on_exception,
                              logfile_name=logfile_name,
                              )
    ift.run(num_fine_tuning_iterations=num_fine_tuning_iterations)


def start_main(args):
    start_iterative_fine_tuning(nickname=args.nickname,
                                planner_params_filename=args.planner_params,
                                classifier_checkpoint=args.classifier_checkpoint,
                                recovery_checkpoint=args.recovery_checkpoint,
                                num_fine_tuning_iterations=args.n_iters,
                                ift_config_filename=args.ift_config,
                                no_execution=args.no_execution,
                                timeout=args.timeout,
                                test_scenes_dir=args.test_scenes_dir,
                                on_exception=args.on_exception,
                                )


def resume_main(args):
    log = load_hjson(args.logfile)
    ift = IterativeFineTuning(log=log,
                              logfile_name=args.logfile,
                              no_execution=args.no_execution,
                              timeout=args.timeout,
                              on_exception=args.on_exception,
                              )
    ift.run(num_fine_tuning_iterations=args.n_iters)


def add_args(start_parser):
    start_parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    start_parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    start_parser.add_argument("--n-iters", '-n', type=int, help='number of iterations of fine tuning', default=100)
    start_parser.add_argument("--no-execution", action="store_true", help='no execution')
    start_parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')


# @notifyme.notify()
@ros_init.with_ros("iterative_fine_tuning")
def ift_main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)
    ou.setLogLevel(ou.LOG_ERROR)
    tf.autograph.set_verbosity(0)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    start_parser = subparsers.add_parser('start')
    resume_parser = subparsers.add_parser('resume')

    start_parser.add_argument("ift_config", type=pathlib.Path, help='hjson file from ift_config/')
    start_parser.add_argument('planner_params', type=pathlib.Path, help='hjson file from planner_configs/')
    start_parser.add_argument("classifier_checkpoint", type=pathlib.Path, help='classifier checkpoint to start from')
    start_parser.add_argument("recovery_checkpoint", type=pathlib.Path, help='recovery checkpoint to start from')
    start_parser.add_argument("nickname", type=str, help='used in making the output directory')
    start_parser.add_argument("test_scenes_dir", type=pathlib.Path)
    start_parser.set_defaults(func=start_main)
    add_args(start_parser)

    resume_parser.add_argument("logfile", type=pathlib.Path)
    resume_parser.set_defaults(func=resume_main)
    add_args(resume_parser)

    # deal_with_exceptions(how_to_handle='retry', function=run_subparsers(parser))
    run_subparsers(parser)


if __name__ == '__main__':
    ift_main()
