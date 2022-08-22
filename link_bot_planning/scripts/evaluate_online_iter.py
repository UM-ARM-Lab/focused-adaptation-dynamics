#!/usr/bin/env python
import argparse
import logging
import os
import pathlib
import time

import colorama
import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_planning.eval_online_utils import evaluate_online_iter_outdir, get_dynamics_and_mde
from link_bot_planning.planning_evaluation import evaluate_multiple_planning, load_planner_params
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.args import int_set_arg
from moonshine.filepath_tools import load_hjson
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)

now = int(time.time())


@ros_init.with_ros(f"planning_evaluation_{now}")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(suppress=True, precision=5, linewidth=250)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params', type=pathlib.Path, help='planner params hjson file')
    parser.add_argument('online_dir', type=pathlib.Path)
    parser.add_argument('iter', type=int)
    parser.add_argument("--scenes", type=pathlib.Path, default=pathlib.Path('test_scenes/car4_alt'))
    parser.add_argument("--trials", type=int_set_arg, default="0-19")
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--yes', '-y', action='store_true')

    args = parser.parse_args()

    online_learning_log = load_hjson(args.online_dir / 'logfile.hjson')

    outdir = evaluate_online_iter_outdir(args.planner_params, args.online_dir, args.iter)

    if outdir.exists() and not args.yes:
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


if __name__ == '__main__':
    import subprocess
    import psutil

    cmd = [
        'roslaunch',
        'link_bot_gazebo',
        'val.launch',
        'world:=car5_alt.world',
        'gui:=false',
        '--no-summary',
    ]
    port = np.random.randint(1_024, 65_000)
    print(f"USING PORTS {port} and {port + 1}")
    os.environ['ROS_MASTER_URI'] = f'http://localhost:{port}'
    os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{port + 1}'
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
    roslaunch_process = psutil.Process(p.pid)

    time.sleep(10)

    print("STARTING MAIN...")
    main()

    for proc in roslaunch_process.children(recursive=True):
        proc.kill()
    roslaunch_process.kill()
