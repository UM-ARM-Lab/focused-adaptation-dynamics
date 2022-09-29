#!/usr/bin/env python
import argparse
import logging
import os
import pathlib
import subprocess
import time

import colorama
import numpy as np
import psutil
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_planning.evaluate_online_iter import evaluate_online_iter
from link_bot_pycommon.args import int_set_arg
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)

now = int(time.time())


@ros_init.with_ros(f"evaluate_online_iter_sim_{now}")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(suppress=True, precision=5, linewidth=250)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params', type=pathlib.Path, help='planner params hjson file')
    parser.add_argument('online_dir', type=pathlib.Path)
    parser.add_argument('iter', type=int)
    parser.add_argument("--scenes", type=pathlib.Path, default=pathlib.Path('test_scenes/car4_alt'))
    parser.add_argument("--trials", type=int_set_arg, default="0-9")
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--yes', '-y', action='store_true')

    args = parser.parse_args()

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
    evaluate_online_iter(planner_params_filename=args.planner_params,
                         online_dir=args.online_dir,
                         iter_idx=args.iter,
                         scenes=args.scenes,
                         trials=args.trials,
                         seed=args.seed,
                         on_exception=args.on_exception,
                         verbose=args.verbose,
                         yes=args.yes,
                         record=False)

    for proc in roslaunch_process.children(recursive=True):
        proc.kill()
    roslaunch_process.kill()


if __name__ == '__main__':
    main()
