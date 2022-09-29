#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import numpy as np
import tensorflow as tf

import rospy
from actionlib_msgs.msg import GoalStatusArray
from arc_utilities import ros_init
from link_bot_classifiers.classifier_utils import strip_torch_model_prefix
from link_bot_planning.planning_evaluation import load_planner_params, evaluate_planning
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.args import int_set_arg
from link_bot_pycommon.load_wandb_model import load_model_artifact
from mde.torch_mde import MDE
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def check_mde_and_dynamics_match(dynamics_run_id, mde_run_id):
    mde_run_id = strip_torch_model_prefix(mde_run_id)
    dynamics_run_id = strip_torch_model_prefix(dynamics_run_id)
    mde = load_model_artifact(mde_run_id, MDE, project='mde', version='best', user='armlab')
    dynamics_used_for_mde = mde.hparams['dataset_hparams']['checkpoint']
    if dynamics_used_for_mde != dynamics_run_id:
        q = input(f"{dynamics_used_for_mde} != {dynamics_run_id} Do you want to override? [y/N]")
        if q != 'y':
            raise RuntimeError()
        print("Ok, continuing...")


@ros_init.with_ros("planning_evaluation")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(suppress=True, precision=5, linewidth=250)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params', type=pathlib.Path, help='planner params hjson file')
    parser.add_argument("test_scenes_dir", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path, help='used in making the output directory')
    parser.add_argument('dynamics', type=pathlib.Path)
    parser.add_argument('mde', type=pathlib.Path)
    parser.add_argument("--trials", type=int_set_arg)
    parser.add_argument("--method-name", type=str)
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--yes", '-y', action='store_true', help='override the dynamics/mde check')
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument("--no-wait-for-move-group", dest="wait_for_move_group", action='store_false')
    parser.set_defaults(wait_for_move_group=True)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    outdir = args.outdir

    if args.test_scenes_dir == pathlib.Path("None"):
        args.test_scenes_dir = None
        assert args.trials is not None
    else:
        if not args.test_scenes_dir.exists():
            print(f"Test scenes dir {args.test_scenes_dir} does not exist")
            return

        if args.trials is None:
            args.trials = list(get_all_scene_indices(args.test_scenes_dir))


    planner_params = load_planner_params(args.planner_params)
    if args.method_name:
        planner_params['method_name'] = args.method_name
    else:
        planner_params['method_name'] = args.outdir.name
    planner_params['fwd_model_dir'] = args.dynamics

    if args.mde == pathlib.Path("None"):
        print("MDE is None!")
        args.mde = None

    if args.mde is None:
        planner_params["classifier_model_dir"] = [pathlib.Path("cl_trials/new_feasibility_baseline/none")]
    else:
        planner_params["classifier_model_dir"] = [args.mde, pathlib.Path("cl_trials/new_feasibility_baseline/none")]

    # NOTE: check that MDE and Dynamics are compatible
    #  - load the MDE
    #  - get the dataset it was trained on
    #  - get the checkpoint used to generate that MDE dataset
    #  - check if it matches the dynamics
    if not args.yes and args.mde is not None:
        check_mde_and_dynamics_match(args.dynamics, args.mde)
    if args.wait_for_move_group:
        rospy.wait_for_message("/hdt_michigan/move_group/status", GoalStatusArray, timeout=10)

    evaluate_planning(outdir=outdir,
                      planner_params=planner_params,
                      trials=args.trials,
                      how_to_handle=args.on_exception,
                      verbose=args.verbose,
                      test_scenes_dir=args.test_scenes_dir,
                      seed=args.seed)

    print("Done!")

if __name__ == '__main__':
    main()
