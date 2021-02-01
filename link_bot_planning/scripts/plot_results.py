#!/usr/bin/env python
import argparse
import json
import pathlib
from typing import Dict

import colorama
import numpy as np

import rospy
from link_bot_planning.my_planner import PlanningQuery
from link_bot_planning.plan_and_execute import TrialStatus
from link_bot_planning.results_utils import labeling_params_from_planner_params, get_paths
from link_bot_pycommon.args import my_formatter, int_set_arg
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import numpify


def main():
    colorama.init(autoreset=True)
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("trial_idx", type=int_set_arg, help='which plan(s) to show')
    parser.add_argument("threshold", type=float)
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--only-timeouts", action='store_true')
    parser.add_argument("--show-tree", action="store_true")
    parser.add_argument("--verbose", '-v', action="count", default=0)

    rospy.init_node("plot_results")

    args = parser.parse_args()

    with (args.results_dir / 'metadata.json').open('r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)
    scenario = get_scenario(metadata['scenario'])

    for trial_idx in args.trial_idx:
        results_filename = args.results_dir / f'{trial_idx}_metrics.pkl.gz'
        datum = load_gzipped_pickle(results_filename)

        should_skip = args.only_timeouts and datum['trial_status'] != TrialStatus.Timeout
        if should_skip:
            continue

        print(f"Trial {trial_idx} ...")
        plot_steps(args.show_tree, scenario, datum, metadata, {'threshold': args.threshold}, args.verbose)
        print(f"... complete with status {datum['trial_status']}")


def get_goal_threshold(planner_params):
    if 'goal_params' in planner_params:
        goal_threshold = planner_params['goal_params']['threshold']
    else:
        goal_threshold = planner_params['goal_threshold']
    return goal_threshold


def plot_steps(show_tree: bool,
               scenario: ExperimentScenario,
               datum: Dict,
               metadata: Dict,
               fallback_labeing_params: Dict,
               verbose: int):
    planner_params = metadata['planner_params']
    goal_threshold = get_goal_threshold(planner_params)

    labeling_params = labeling_params_from_planner_params(planner_params, fallback_labeing_params)

    steps = datum['steps']

    if len(steps) == 0:
        q: PlanningQuery = datum['planning_queries'][0]
        start = q.start
        goal = q.goal
        environment = q.environment
        anim = RvizAnimationController(n_time_steps=1)
        scenario.plot_state_rviz(start, label='actual', color='#ff0000aa')
        scenario.plot_goal_rviz(goal, goal_threshold)
        scenario.plot_environment_rviz(environment)
        anim.step()
        return

    goal = datum['goal']
    first_step = steps[0]
    planning_query: PlanningQuery = first_step['planning_query']
    environment = numpify(planning_query.environment)
    actions, actual_states, predicted_states, types = get_paths(datum, scenario, show_tree, verbose)

    anim = RvizAnimationController(n_time_steps=len(actual_states))

    def _type_action_color(type_t: str):
        if type_t == 'executed_plan':
            return 'b'
        elif type_t == 'executed_recovery':
            return '#ff00ff'

    scenario.reset_planning_viz()
    while not anim.done:
        scenario.plot_environment_rviz(environment)
        t = anim.t()
        s_t = actual_states[t]
        s_t_pred = predicted_states[t]
        scenario.plot_state_rviz(s_t, label='actual', color='#ff0000aa')
        c = '#0000ffaa'
        if len(actions) > 0:
            if t < anim.max_t:
                type_t = types[t]
                action_color = _type_action_color(type_t)
                scenario.plot_action_rviz(s_t, actions[t], color=action_color)
            else:
                type_t = types[t - 1]
                action_color = _type_action_color(type_t)
                scenario.plot_action_rviz(actual_states[t - 1], actions[t - 1], color=action_color)

        if s_t_pred is not None:
            scenario.plot_state_rviz(s_t_pred, label='predicted', color=c)
            is_close = scenario.compute_label(s_t, s_t_pred, labeling_params)
            scenario.plot_is_close(is_close)
        else:
            scenario.plot_is_close(None)
        dist_to_goal = scenario.distance_to_goal(s_t, goal)
        actually_at_goal = dist_to_goal < goal_threshold
        scenario.plot_goal_rviz(goal, goal_threshold, actually_at_goal)
        anim.step()


if __name__ == '__main__':
    main()
