#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import pathlib
import time
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from colorama import Fore
from ompl import base as ob

from link_bot_data import random_environment_data_utils
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning import my_mpc
from link_bot_planning.ompl_viz import plot
from link_bot_planning.params import PlannerParams, LocalEnvParams, EnvParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


class TestWithClassifier(my_mpc.myMPC):

    def __init__(self,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 classifier_model_dir: pathlib.Path,
                 classifier_model_type: str,
                 n_targets: int,
                 n_envs: int,
                 verbose: int,
                 planner_params: PlannerParams,
                 local_env_params: LocalEnvParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 seed: int,
                 outdir: Optional[pathlib.Path] = None,
                 ):
        super().__init__(fwd_model_dir=fwd_model_dir,
                         fwd_model_type=fwd_model_type,
                         classifier_model_dir=classifier_model_dir,
                         classifier_model_type=classifier_model_type,
                         n_envs=n_envs,
                         n_targets_per_env=n_targets,
                         verbose=verbose,
                         planner_params=planner_params,
                         local_env_params=local_env_params,
                         env_params=env_params,
                         services=services,
                         no_execution=False)
        self.outdir = outdir
        self.seed = seed

        self.metrics = {
            "fwd_model_dir": str(fwd_model_dir),
            "fwd_model_type": fwd_model_type,
            "classifier_model_dir": str(classifier_model_dir),
            "classifier_model_type": classifier_model_type,
            "n_envs": n_targets,
            "n_targets": n_targets,
            "planner_params": planner_params.to_json(),
            "local_env_params": local_env_params.to_json(),
            "env_params": env_params.to_json(),
            "seed": self.seed,
            "metrics": [],
        }
        self.root = self.outdir / self.classifier_model_type
        self.root.mkdir(parents=True)
        print(Fore.CYAN + str(self.root) + Fore.RESET)
        self.metrics_filename = self.root / 'metrics.json'
        self.successfully_completed_plan_idx = 0

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_sdf_data: link_bot_sdf_utils.SDF,
                         planner_data: ob.PlannerData,
                         planning_time: float):
        final_error = np.linalg.norm(planned_path[-1, 0:2] - tail_goal_point)
        lengths = [np.linalg.norm(planned_path[i] - planned_path[i - 1]) for i in range(1, len(planned_path))]
        path_length = np.sum(lengths)
        duration = self.fwd_model.dt * len(planned_path)

        print(self.successfully_completed_plan_idx)

        metrics_for_plan = {
            'planning_time': planning_time,
            'final_error': final_error,
            'path_length': path_length,
        }
        self.metrics['metrics'].append(metrics_for_plan)
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file)

        full_binary = full_sdf_data.sdf > 0
        plot(self.viz_object, planner_data, full_binary, tail_goal_point, planned_path, planned_actions, full_sdf_data.extent)
        plan_viz_path = self.root / "plan_{}.png".format(self.successfully_completed_plan_idx)
        plt.savefig(plan_viz_path)

        if self.verbose >= 1:
            msg = "Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s"
            print(msg.format(final_error, path_length, len(planned_path), duration))
            plt.show()

        self.successfully_completed_plan_idx += 1

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              planner_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_path: np.ndarray):
        pass


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="forward model", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument("classifier_1_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument("classifier_1_model_type", choices=['none', 'collision', 'raster'])
    parser.add_argument("classifier_2_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument("classifier_2_model_type", choices=['none', 'collision', 'raster'])
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--n-targets", type=int, default=10, help='number of targets/plans per env')
    parser.add_argument("--n-envs", type=int, default=10, help='number of envs')
    parser.add_argument("--seed", '-s', type=int, default=3)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=15.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0, help='real time rate')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=5, help='environment width')
    parser.add_argument('--env-h', type=float, default=5, help='environment height')
    parser.add_argument('--local-env-cols', type=float, default=100, help='local env width')
    parser.add_argument('--local-env-rows', type=float, default=100, help='local env width')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    now = str(int(time.time()))
    common_output_directory = random_environment_data_utils.data_directory(args.outdir, now)
    common_output_directory = pathlib.Path(common_output_directory)
    if not common_output_directory.is_dir():
        print(Fore.YELLOW + "Creating output directory: {}".format(common_output_directory) + Fore.RESET)
        common_output_directory.mkdir(parents=True)

    planner_params = PlannerParams(timeout=args.planner_timeout, max_v=args.max_v)
    local_env_params = LocalEnvParams(h_rows=args.local_env_rows,
                                      w_cols=args.local_env_cols,
                                      res=args.res)
    env_params = EnvParams(w=args.env_w,
                           h=args.env_h,
                           real_time_rate=args.real_time_rate,
                           goal_padding=0.0)

    rospy.init_node('test_planner_with_classifier')

    initial_object_dict = {
        'moving_box1': [2.0, 0],
        'moving_box2': [-1.5, 0],
        'moving_box3': [-0.5, 1],
        'moving_box4': [1.5, - 2],
        'moving_box5': [-1.5, - 2.0],
        'moving_box6': [-0.5, 2.0],
    }

    services = gazebo_utils.setup_gazebo_env(verbose=args.verbose,
                                             real_time_rate=env_params.real_time_rate,
                                             reset_world=True,
                                             initial_object_dict=initial_object_dict)
    services.pause(std_srvs.srv.EmptyRequest())

    classifier_1_tester = TestWithClassifier(
        fwd_model_dir=args.fwd_model_dir,
        fwd_model_type=args.fwd_model_type,
        classifier_model_dir=args.classifier_1_model_dir,
        classifier_model_type=args.classifier_1_model_type,
        n_targets=args.n_targets,
        n_envs=args.n_envs,
        verbose=args.verbose,
        planner_params=planner_params,
        local_env_params=local_env_params,
        env_params=env_params,
        services=services,
        seed=args.seed,
        outdir=common_output_directory,
    )
    classifier_1_tester.run()

    # Reset everything
    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    services = gazebo_utils.setup_gazebo_env(verbose=args.verbose,
                                             real_time_rate=env_params.real_time_rate,
                                             reset_world=True,
                                             initial_object_dict=initial_object_dict)
    services.pause(std_srvs.srv.EmptyRequest())

    classifier_2_tester = TestWithClassifier(
        fwd_model_dir=args.fwd_model_dir,
        fwd_model_type=args.fwd_model_type,
        classifier_model_dir=args.classifier_2_model_dir,
        classifier_model_type=args.classifier_2_model_type,
        n_targets=args.n_targets,
        n_envs=args.n_envs,
        verbose=args.verbose,
        planner_params=planner_params,
        local_env_params=local_env_params,
        env_params=env_params,
        services=services,
        seed=args.seed,
        outdir=common_output_directory,
    )
    classifier_2_tester.run()


if __name__ == '__main__':
    main()
