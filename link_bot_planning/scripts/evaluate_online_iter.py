#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_planning.planning_evaluation import evaluate_multiple_planning, load_planner_params
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.args import int_set_arg
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def get_dynamics_and_mde(log, i: int):
    iter_log = log[f'iter{i}']
    dynamics_run_id = iter_log['dynamics_run_id']
    mde_run_id = iter_log.get('mde_run_id', None)
    if mde_run_id is None:
        return f'p:{dynamics_run_id}', None
    else:
        return f'p:{dynamics_run_id}', f'p:{mde_run_id}'


@ros_init.with_ros("planning_evaluation")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(suppress=True, precision=5, linewidth=250)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params', type=pathlib.Path, help='planner params hjson file')
    parser.add_argument('online_dir', type=pathlib.Path)
    parser.add_argument('iter', type=int)
    parser.add_argument("--scenes", type=pathlib.Path, default=pathlib.Path('test_scenes/car4_alt'))
    parser.add_argument("--trials", type=int_set_arg)
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")

    args = parser.parse_args()

    online_learning_log = load_hjson(args.online_dir / 'logfile.hjson')

    outdir = evaluate_online_iter_outdir(args.planner_params, args.online_dir)

    if outdir.exists():
        k = input(f"{outdir.as_posix()} exists, do you want to resume? [Y/n]")
        if k in ['n', 'N']:
            print("Aborting")
            return

    dynamics, mde = get_dynamics_and_mde(online_learning_log, args.iter)

    planner_params = load_planner_params(args.planner_params)
    planner_params['online_iter'] = args.iter
    planner_params['method_name'] = outdir.name
    planner_params['fwd_model_dir'] = dynamics
    if mde is None:
        planner_params["classifier_model_dir"] = [pathlib.Path("cl_trials/new_feasibility_baseline/none")]
    else:
        planner_params["classifier_model_dir"] = [mde, pathlib.Path("cl_trials/new_feasibility_baseline/none")]

    if not args.scenes.exists():
        print(f"Test scenes dir {args.scenes} does not exist")
        return

    if args.trials is None:
        args.trials = list(get_all_scene_indices(args.scenes))
        print(args.trials)

    evaluate_multiple_planning(outdir=outdir,
                               planners_params=[(args.planner_params.stem, planner_params)],
                               trials=args.trials,
                               how_to_handle=args.on_exception,
                               verbose=args.verbose,
                               test_scenes_dir=args.scenes,
                               seed=args.seed)


def evaluate_online_iter_outdir(planner_params: pathlib.Path, online_dir: pathlib.Path):
    planning_params_name = planner_params.stem
    nickname = f"{online_dir.name}_iter{iter}-{planning_params_name}"
    outdir = pathlib.Path(f"/media/shared/planning_results/{nickname}")
    return outdir


if __name__ == '__main__':
    main()
