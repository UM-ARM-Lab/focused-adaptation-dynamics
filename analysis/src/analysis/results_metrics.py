import logging
import pathlib
from typing import Dict, Optional

import numpy as np
import rospkg

from analysis.results_utils import get_paths, try_load_classifier_params
from arc_utilities.algorithms import nested_dict_update
from link_bot_planning.my_planner import PlanningResult, MyPlannerStatus
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.func_list_registrar import FuncListRegistrar
from link_bot_pycommon.pycommon import has_keys, paths_from_json
from moonshine.filepath_tools import load_hjson
from moonshine.numpify import numpify

logger = logging.getLogger(__file__)

metrics_funcs = FuncListRegistrar()


@metrics_funcs
def full_retrain(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    return has_keys(trial_metadata, ['ift_config', 'full_retrain_classifier'])


@metrics_funcs
def ift_uuid(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    default_value = 'no_ift_uuid'
    uuid = trial_metadata.get('ift_uuid', default_value)
    return uuid


@metrics_funcs
def starts_with_recovery(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    try:
        first_step = trial_datum['steps'][0]
        return first_step['type'] == 'executed_recovery'
    except Exception:
        return False


@metrics_funcs
def num_recovery_actions(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    count = 0
    for step in trial_datum['steps']:
        if step['type'] == 'executed_recovery':
            count += 1
    return count


@metrics_funcs
def num_steps(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    paths = list(get_paths(trial_datum))
    return len(paths)


@metrics_funcs
def mean_accept_probability(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    total = 0
    n_actions = 0
    for _, _, actual_state_t, planned_state_t, _, _ in get_paths(trial_datum):
        if planned_state_t is not None and 'accept_probability' in planned_state_t:
            p = planned_state_t['accept_probability']
            total += p
            n_actions += 1

    if n_actions == 0:
        return 1

    return total / n_actions


@metrics_funcs
def mean_error_accept_agreement(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    n_actions = 0
    total = 0
    for _, _, actual_state_t, planned_state_t, _, _ in get_paths(trial_datum):
        if planned_state_t is not None and 'accept_probability' in planned_state_t:
            d = scenario.classifier_distance(actual_state_t, planned_state_t)
            p = planned_state_t['accept_probability']
            total += 1 - abs(np.exp(-d) - p)
            n_actions += 1

    if n_actions == 0:
        return 1
    return total / n_actions


# @metrics_funcs
# def cumulative_task_error(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
#     goal = trial_datum['goal']
#     cumulative_error = 0
#     for _, _, actual_state_t, _, _, _ in get_paths(trial_datum):
#         cumulative_error += numpify(scenario.distance_to_goal(actual_state_t, goal))
#     return cumulative_error
#
#
# @metrics_funcs
# def cumulative_planning_error(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
#     goal = trial_datum['goal']
#     cumulative_error = 0
#     for _, _, actual_state_t, _, _, _ in get_paths(trial_datum, full_path=True):
#         cumulative_error += numpify(scenario.distance_to_goal(actual_state_t, goal))
#     return cumulative_error


@metrics_funcs
def min_planning_error(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    final_plan_to_goal_errors = []
    for step in trial_datum['steps']:
        final_planned_state = step['planning_result'].path[-1]
        final_plan_to_goal_error = scenario.distance_to_goal(final_planned_state, goal)
        final_plan_to_goal_errors.append(final_plan_to_goal_error)
    min_final_plan_to_goal_error = np.min(final_plan_to_goal_errors)
    return min_final_plan_to_goal_error


@metrics_funcs
def min_task_error(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    final_execution_to_goal_errors = []
    for step in trial_datum['steps']:
        final_actual_state = step['execution_result'].path[-1]
        final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
        final_execution_to_goal_errors.append(final_execution_to_goal_error)
    min_final_execution_to_goal_error = np.min(final_execution_to_goal_errors)
    return min_final_execution_to_goal_error


@metrics_funcs
def min_error_discrepancy(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    min_final_execution_to_goal_error = min_task_error(_, scenario, __, trial_datum)
    min_final_plan_to_goal_error = min_planning_error(_, scenario, __, trial_datum)
    return abs(min_final_execution_to_goal_error - min_final_plan_to_goal_error)


@metrics_funcs
def combined_error(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    min_final_execution_to_goal_error = min_task_error(_, scenario, __, trial_datum)
    min_final_plan_to_goal_error = min_planning_error(_, scenario, __, trial_datum)
    nme = normalized_model_error(_, scenario, __, trial_datum)
    return (min_final_plan_to_goal_error + min_final_execution_to_goal_error + nme) / 3


@metrics_funcs
def num_failed_actions(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    failed_actions = 0
    for _, _, actual_state_t, planned_state_t, type_t, _ in get_paths(trial_datum, False):
        if planned_state_t is not None and actual_state_t is not None:
            if scenario.simple_name() == "watering":
                controlled_container_pos_error = np.linalg.norm(planned_state_t['controlled_container_pos'] - actual_state_t['controlled_container_pos'])
                target_volume_error = np.linalg.norm(planned_state_t['target_volume'] - actual_state_t['target_volume'])
                if controlled_container_pos_error > 0.1 or target_volume_error > 0.3:
                    failed_actions += 1

            else:
                left_gripper_error = np.linalg.norm(planned_state_t['left_gripper'] - actual_state_t['left_gripper'])
                right_gripper_error = np.linalg.norm(planned_state_t['right_gripper'] - actual_state_t['right_gripper'])
                if left_gripper_error > 0.05 or right_gripper_error > 0.05:
                    failed_actions += 1
    return failed_actions


@metrics_funcs
def task_error(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    goal = trial_datum['goal']
    final_actual_state = trial_datum['end_state']
    final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)
    return numpify(final_execution_to_goal_error)


@metrics_funcs
def accept_type(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    return trial_metadata['planner_params'].get('accept_type', 'strict')


@metrics_funcs
def timeout(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    return trial_metadata['planner_params']['termination_criteria']['timeout']


@metrics_funcs
def stop_on_error(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    soe = has_keys(trial_metadata, ['planner_params', 'stop_on_error_above'], 999)
    return soe < 1


@metrics_funcs
def task_error_given_solved(path: pathlib.Path, scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    if any_solved(path, scenario, trial_metadata, trial_datum):
        return task_error(path, scenario, trial_metadata, trial_datum)
    else:
        return np.NAN


@metrics_funcs
def success_given_solved(path: pathlib.Path, scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    if any_solved(path, scenario, trial_metadata, trial_datum):
        return success(path, scenario, trial_metadata, trial_datum)
    else:
        return np.NAN


@metrics_funcs
def success(path: pathlib.Path, scenario: ExperimentScenario, trial_metadata: Dict, trial_datum: Dict):
    final_execution_to_goal_error = task_error(path, scenario, trial_metadata, trial_datum)
    return int(final_execution_to_goal_error < trial_metadata['planner_params']['goal_params']['threshold'])


@metrics_funcs
def recovery_success(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    recovery_started = False
    recoveries_finished = 0
    recoveries_started = 0
    for i, step in enumerate(trial_datum['steps']):
        if recovery_started and step['type'] != 'executed_recovery':
            recoveries_finished += 1
            recovery_started = False
        elif step['type'] == 'executed_recovery' and not recovery_started:
            recoveries_started += 1
            recovery_started = True
    if recoveries_started == 0:
        _recovery_success = np.nan
    else:
        _recovery_success = np.divide(recoveries_finished, recoveries_started)
    return _recovery_success


@metrics_funcs
def planning_time(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    _planning_time = 0
    for step in trial_datum['steps']:
        _planning_time += step['planning_result'].time
    return _planning_time


@metrics_funcs
def max_planning_time(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    planning_times = []
    for step in trial_datum['steps']:
        planning_times.append(step['planning_result'].time)
    return max(planning_times)


@metrics_funcs
def extensions_per_second(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    attempted_extensions = []
    for step in trial_datum['steps']:
        attempted_extensions.append(step['planning_result'].attempted_extensions)
    total_num_extensions = sum(attempted_extensions)
    planning_times = []
    for step in trial_datum['steps']:
        planning_times.append(step['planning_result'].time)
    total_planning_time = sum(planning_times)
    avg_extensions_per_second = total_num_extensions / total_planning_time
    return avg_extensions_per_second


@metrics_funcs
def total_extensions(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    attempted_extensions = []
    for step in trial_datum['steps']:
        attempted_extensions.append(step['planning_result'].attempted_extensions)
    total_num_extensions = sum(attempted_extensions)
    return total_num_extensions


@metrics_funcs
def max_extensions(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    attempted_extensions = []
    for step in trial_datum['steps']:
        attempted_extensions.append(step['planning_result'].attempted_extensions)
    max_num_extensions = max(attempted_extensions)
    return max_num_extensions


@metrics_funcs
def mean_progagation_time(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    progagation_times = []
    # average across all the planning results in the trial
    for step in trial_datum['steps']:
        if 'planning_result' in step:
            dt = step['planning_result'].mean_propagate_time
            if dt is None:
                dt = np.nan
            progagation_times.append(dt)
    if len(progagation_times) == 0:
        return np.nan
    else:
        return np.mean(progagation_times)


@metrics_funcs
def total_time(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    _total_time = trial_datum['total_time']
    return _total_time


@metrics_funcs
def num_planning_attempts(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    attempts = 0
    for step in trial_datum['steps']:
        if step['type'] == 'executed_plan':
            attempts += 1
    return attempts


@metrics_funcs
def any_solved(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    solved = False
    for step in trial_datum['steps']:
        if step['type'] == 'executed_plan':
            planning_result: PlanningResult = step['planning_result']
            if planning_result.status == MyPlannerStatus.Solved:
                solved = True
    return solved


@metrics_funcs
def num_trials(_: pathlib.Path, __: ExperimentScenario, ___: Dict, ____: Dict):
    return 1


@metrics_funcs
def num_actions(_: pathlib.Path, __: ExperimentScenario, ___: Dict, trial_datum: Dict):
    return len(list(get_paths(trial_datum)))


@metrics_funcs
def normalized_model_error(_: pathlib.Path, scenario: ExperimentScenario, __: Dict, trial_datum: Dict):
    total_model_error = 0.0
    n_total_actions = 0
    for _, _, actual_state_t, planned_state_t, type_t, _ in get_paths(trial_datum):
        if planned_state_t is not None:
            model_error = scenario.classifier_distance(actual_state_t, planned_state_t)
            total_model_error += model_error
            n_total_actions += 1
    if n_total_actions == 0:
        return np.NAN
    return total_model_error / n_total_actions


@metrics_funcs
def max_extensions_param(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    return trial_metadata['planner_params']['termination_criteria']['max_extensions']


@metrics_funcs
def online_iter(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    return trial_metadata['planner_params'].get('online_iter', None)


@metrics_funcs
def max_attempts(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    return trial_metadata['planner_params']['termination_criteria']['max_attempts']


@metrics_funcs
def max_planning_attempts(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    return trial_metadata['planner_params']['termination_criteria']['max_planning_attempts']


@metrics_funcs
def recovery_name(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    r = trial_metadata['planner_params']['recovery']
    use_recovery = r.get('use_recovery', False)
    if not use_recovery:
        return 'no-recovery'
    recovery_model_dir = r["recovery_model_dir"]
    return pathlib.Path(*pathlib.Path(recovery_model_dir).parent.parts[-2:]).as_posix()


@metrics_funcs
def classifier_name(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    c = trial_metadata['planner_params']['classifier_model_dir']
    found = False
    classifier_name_ = None
    for c_i in c:
        if 'best_checkpoint' in c_i:
            if found:
                logger.warning("Multiple learned classifiers detected!!!")
            found = True
            classifier_name_ = pathlib.Path(*pathlib.Path(c_i).parent.parts[-2:]).as_posix()
    if not found:
        if len(c) >= 1:
            classifier_name_ = c[0]
            found = True
        elif len(c) == 0:
            classifier_name_ = 'no classifier'
            found = True
    if not found:
        raise RuntimeError(f"Could not guess the classifier name:\n{c}")

    return classifier_name_


@metrics_funcs
def classifier_dataset(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    try:
        classifier_model_dirs = paths_from_json(trial_metadata['planner_params']['classifier_model_dir'])
        for representative_classifier_model_dir in classifier_model_dirs:
            if 'checkpoint' in representative_classifier_model_dir.as_posix():
                return representative_classifier_model_dir
        return "no-learned-classifiers"
    except RuntimeError:
        return None


@metrics_funcs
def target_env(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    if trial_metadata['test_scenes_dir'] is None:
        return "no_test_scene_dir"
    return pathlib.Path(trial_metadata['test_scenes_dir']).name


@metrics_funcs
def dmax(_: pathlib.Path, __: ExperimentScenario, trial_metadata: Dict, ___: Dict):
    pp = trial_metadata['planner_params']
    if 'mde_threshold' in pp:
        return pp['mde_threshold']
    elif 'dmax' in pp:
        return pp['dmax']
    else:
        return -1


def load_analysis_params(analysis_params_filename: Optional[pathlib.Path] = None):
    analysis_params = load_analysis_hjson(pathlib.Path("analysis_params/common.json"))

    if analysis_params_filename is not None:
        analysis_params = nested_dict_update(analysis_params, load_hjson(analysis_params_filename))

    return analysis_params


def load_analysis_hjson(analysis_params_filename: pathlib.Path):
    r = rospkg.RosPack()
    analysis_dir = pathlib.Path(r.get_path("analysis"))
    analysis_params_common_filename = analysis_dir / analysis_params_filename
    analysis_params = load_hjson(analysis_params_common_filename)
    return analysis_params


def trial_metadata_to_classifier_hparams(path, trial_metadata):
    classifier_model_dir = trial_metadata['planner_params']['classifier_model_dir']
    if isinstance(classifier_model_dir, list):
        classifier_model_dir = classifier_model_dir[0]
    classifier_model_dir = pathlib.Path(classifier_model_dir)
    classifier_hparams = try_load_classifier_params(classifier_model_dir, path.parent.parent.parent)
    return classifier_hparams


metrics_names = [func.__name__ for func in metrics_funcs]
