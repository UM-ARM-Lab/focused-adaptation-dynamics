import pathlib
import socket
import uuid
import warnings
from time import time, sleep
from typing import Optional, Dict, List, Tuple, Callable

import numpy as np
from colorama import Fore

from link_bot_gazebo.gazebo_utils import get_gazebo_processes
from link_bot_planning.trial_result import planning_trial_name
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou

import rospy
from arc_utilities.algorithms import nested_dict_update
from link_bot_data.dataset_utils import git_sha
from link_bot_planning import plan_and_execute
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.pycommon import deal_with_exceptions, empty_callable
from link_bot_pycommon.get_service_provider import get_service_provider
from link_bot_pycommon.serialization import dump_gzipped_pickle, my_hdump
from moonshine.filepath_tools import load_hjson
from moonshine.numpify import numpify


def get_results_filename(outdir: pathlib.Path, trial_idx: int):
    data_filename = planning_trial_name(trial_idx)
    full_data_filename = outdir / data_filename
    return full_data_filename


class EvaluatePlanning(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 service_provider: BaseServices,
                 verbose: int,
                 planner_params: Dict,
                 outdir: pathlib.Path,
                 trials: Optional[List[int]] = None,
                 record: Optional[bool] = False,
                 no_execution: Optional[bool] = False,
                 test_scenes_dir: Optional[pathlib.Path] = None,
                 extra_end_conditions: Optional[List[Callable]] = None,
                 metadata_update: Optional[Dict] = None,
                 seed: int = 0,
                 recovery_seed: int = 0,
                 ):
        super().__init__(planner, trials=trials, verbose=verbose, planner_params=planner_params,
                         service_provider=service_provider, no_execution=no_execution,
                         test_scenes_dir=test_scenes_dir, seed=seed, extra_end_conditions=extra_end_conditions,
                         recovery_seed=recovery_seed)
        self.record = record
        self.outdir = outdir

        self.outdir.mkdir(parents=True, exist_ok=True)
        rospy.loginfo(Fore.BLUE + f"Output directory: {self.outdir.as_posix()}")

        if self.test_scenes_dir is None:
            scene_name = None
        else:
            scene_name = self.test_scenes_dir.name.replace("_", " ")
        metadata = {
            "trials":                self.trials,
            "planner_params":        self.planner_params,
            "scenario":              self.planner.scenario.simple_name(),
            "commit":                git_sha(),
            "scene_name":            scene_name,
            "test_scenes_dir":       self.test_scenes_dir.as_posix() if self.test_scenes_dir is not None else None,
            'hostname':              socket.gethostname(),
            'seed':                  self.seed,
            'experiment_start_time': int(time()),
            'experiment_uuid':       uuid.uuid4(),
            'world_file_name':       rospy.get_param('world_file_name', None),
            'world_initial_sdf':     service_provider.get_world_initial_sdf(),
        }
        metadata.update(self.planner.get_metadata())
        if metadata_update is not None:
            metadata.update(metadata_update)
        with (self.outdir / 'metadata.hjson').open("w") as metadata_file:
            my_hdump(metadata, metadata_file, indent=2)

        self.bag = None
        self.final_execution_to_goal_errors = []
        self.trial_times = []

    def randomize_environment(self):
        if self.verbose >= 1:
            rospy.loginfo("Randomizing env")
        self.service_provider.play()
        super().randomize_environment()
        self.service_provider.pause()
        if self.verbose >= 1:
            rospy.loginfo("End randomizing env")

    def on_before_action(self, trial_idx, attempt_idx):
        if self.record:
            self.service_provider.stop_record_trial()
            filename = f"{trial_idx:04d}-{attempt_idx:04d}-{int(time())}.mp4"
            filename = pathlib.Path('/media/shared/captures/icra_2023') / self.outdir / filename
            filename.parent.mkdir(exist_ok=True, parents=True)
            self.service_provider.start_record_trial(str(filename))

    def on_after_action(self):
        if self.record:
            sleep(1)
            self.service_provider.stop_record_trial()

    def on_trial_complete(self, trial_data: Dict, trial_idx: int):
        extra_trial_data = {
            "planner_params": self.planner_params,
            "scenario":       self.planner_params['scenario'],
            'current_time':   int(time()),
            'uuid':           uuid.uuid4(),
        }
        trial_data.update(extra_trial_data)
        full_data_filename = get_results_filename(self.outdir, trial_idx)
        dump_gzipped_pickle(trial_data, full_data_filename)

        # print some useful information
        goal = trial_data['planning_queries'][0].goal
        final_actual_state = numpify(trial_data['end_state'])
        final_execution_to_goal_error = self.planner.scenario.distance_to_goal(final_actual_state, goal)
        trial_time = trial_data['total_time']
        self.final_execution_to_goal_errors.append(final_execution_to_goal_error)
        self.trial_times.append(trial_time)
        goal_threshold = self.planner_params['goal_params']['threshold']
        n = len(self.final_execution_to_goal_errors)
        n_success = np.count_nonzero(np.array(self.final_execution_to_goal_errors) < goal_threshold)
        success_percentage = n_success / n * 100
        current_mean_error = np.mean(np.array(self.final_execution_to_goal_errors))
        current_mean_trial_time = np.mean(np.array(self.trial_times))
        update_msg = [
            f"Success={success_percentage:.2f}% [{n_success}/{n}]",
            f"Mean Error={current_mean_error:.3f}",
            f"Mean Trial Time={current_mean_trial_time:.3f}s",
        ]
        rospy.loginfo(Fore.LIGHTBLUE_EX + f"[{self.outdir.name}] " + Fore.RESET + ', '.join(update_msg))

    def plan_and_execute(self, trial_idx: int):
        full_data_filename = get_results_filename(self.outdir, trial_idx)
        if full_data_filename.exists():
            rospy.loginfo(f"Found existing trial {trial_idx}, skipping.")
            return
        super().plan_and_execute(trial_idx=trial_idx)

    def setup_test_scene(self, trial_idx: int):
        if self.record:
            self.service_provider.stop_record_trial()
            filename = pathlib.Path('/media/shared/captures') / self.outdir / f"{trial_idx:04d}-reset.mp4"
            filename.parent.mkdir(exist_ok=True, parents=True)
            self.service_provider.start_record_trial(str(filename))
        super().setup_test_scene(trial_idx)
        if self.record:
            self.service_provider.stop_record_trial()

    def on_complete(self):
        super().on_complete()


def evaluate_planning(planner_params: Dict,
                      outdir: pathlib.Path,
                      scenario: ScenarioWithVisualization = None,
                      trials: Optional[List[int]] = None,
                      verbose: int = 0,
                      record: bool = False,
                      no_execution: bool = False,
                      timeout: Optional[int] = None,
                      test_scenes_dir: Optional[pathlib.Path] = None,
                      seed: int = 0,
                      recovery_seed: int = 0,
                      log_full_tree: bool = True,
                      how_to_handle: str = 'retry',
                      on_scenario_cb=empty_callable,
                      ):
    # first check if the outputs already exist
    done = True
    for trial_idx in trials:
        full_data_filename = get_results_filename(outdir, trial_idx)
        if not full_data_filename.exists():
            done = False
    if done:
        print("Nothing to do!")
        return outdir

    # override some arguments
    if timeout is not None:
        rospy.loginfo(f"Overriding with timeout {timeout}")
        planner_params["termination_criteria"]['timeout'] = timeout
        planner_params["termination_criteria"]['total_timeout'] = timeout
    planner_params["log_full_tree"] = log_full_tree

    # ensure gazebo processes are not suspended
    gazebo_processes = get_gazebo_processes()
    [p.resume() for p in gazebo_processes]

    # Start Services
    service_provider = get_service_provider(planner_params.get('service_provider', 'gazebo'))
    service_provider.play()  # time needs to be advancing while we setup the planner so it can use ROS to query things
    planner = get_planner(planner_params=planner_params, verbose=verbose, log_full_tree=log_full_tree,
                          scenario=scenario)
    on_scenario_cb(planner.scenario)

    service_provider.setup_env(verbose=verbose,
                               real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size,
                               play=True)

    planner.scenario.on_before_get_state_or_execute_action()

    runner = EvaluatePlanning(planner=planner,
                              service_provider=service_provider,
                              trials=trials,
                              verbose=verbose,
                              planner_params=planner_params,
                              outdir=outdir,
                              record=record,
                              no_execution=no_execution,
                              test_scenes_dir=test_scenes_dir,
                              seed=seed,
                              recovery_seed=recovery_seed,
                              )

    def _on_exception():
        pass

    deal_with_exceptions(how_to_handle=how_to_handle,
                         function=runner.run,
                         exception_callback=_on_exception,
                         )
    planner.scenario.robot.disconnect()


def evaluate_multiple_planning(outdir: pathlib.Path,
                               planners_params: List[Tuple[str, Dict]],
                               logfile_name: Optional = None,
                               trials: Optional[List[int]] = None,
                               start_idx: int = 0,
                               stop_idx: int = -1,
                               how_to_handle: Optional[str] = 'raise',
                               verbose: int = 0,
                               record: bool = False,
                               no_execution: bool = False,
                               timeout: Optional[int] = None,
                               test_scenes_dir: Optional[pathlib.Path] = None,
                               seed: int = 0,
                               log_full_tree: bool = True,
                               on_scenario_cb=empty_callable,
                               ):
    ou.setLogLevel(ou.LOG_ERROR)

    if logfile_name is None:
        logfile_name = outdir / f'logfile.hjson'

    print(f'logfile: {logfile_name}')

    rospy.loginfo(Fore.CYAN + "common output directory: {}".format(outdir))
    if not outdir.is_dir():
        rospy.loginfo(Fore.YELLOW + "Creating output directory: {}".format(outdir))
        outdir.mkdir(parents=True)

    # NOTE: if method names are not unique, we would overwrite results. Very bad!
    planners_params = ensure_unique_method_name(planners_params)

    for comparison_idx, (method_name, planner_params) in enumerate(planners_params):
        if comparison_idx < start_idx:
            continue
        if stop_idx != -1 and comparison_idx >= stop_idx:
            break

        rospy.loginfo(Fore.GREEN + f"Running method {method_name}")

        evaluate_planning(planner_params=planner_params,
                          trials=trials,
                          outdir=outdir,
                          verbose=verbose,
                          record=record,
                          no_execution=no_execution,
                          timeout=timeout,
                          test_scenes_dir=test_scenes_dir,
                          seed=seed,
                          log_full_tree=log_full_tree,
                          how_to_handle=how_to_handle,
                          on_scenario_cb=on_scenario_cb,
                          )

        rospy.loginfo(f"Results written to {outdir}")

    return outdir


def ensure_unique_method_name(planners_params):
    unique_method_params = []
    for original_method_name, params in planners_params:
        d = 1
        method_name = original_method_name
        while method_name in [n for n, _ in unique_method_params]:
            method_name = original_method_name + f"_{d}"
            d += 1
        if original_method_name != method_name:
            rospy.logwarn(f"Making method name {original_method_name} unique -> {method_name}")
        unique_method_params.append((method_name, params))
    return unique_method_params


def load_planner_params(filename: pathlib.Path):
    top_level_common_filename = filename.parent.parent / 'common.hjson'
    top_level_common_params = load_hjson(top_level_common_filename)

    common_filename = filename.parent / 'common.hjson'
    common_params = load_hjson(common_filename)

    params = load_hjson(filename)
    common_params = nested_dict_update(top_level_common_params, common_params)
    params = nested_dict_update(common_params, params)
    return params
