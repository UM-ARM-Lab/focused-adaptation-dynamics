#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_planning.evaluate_online_iter import evaluate_online_iter
from link_bot_pycommon.args import int_set_arg
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros(f"evaluate_online_iter_real")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(suppress=True, precision=5, linewidth=250)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('online_dir', type=pathlib.Path)
    parser.add_argument('iter', type=int)
    parser.add_argument('--planner-params', type=pathlib.Path,
                        default=pathlib.Path("planner_configs/val_car/real_soe.hjson"))
    parser.add_argument("--scenes", type=pathlib.Path, default=pathlib.Path('test_scenes/car5_real'))
    parser.add_argument("--trials", type=int_set_arg, default="0-19")
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--yes', '-y', action='store_true')

    args = parser.parse_args()

    evaluate_online_iter(planner_params_filename=args.planner_params,
                         online_dir=args.online_dir,
                         iter_idx=args.iter,
                         scenes=args.scenes,
                         trials=args.trials,
                         seed=args.seed,
                         on_exception=args.on_exception,
                         verbose=args.verbose,
                         yes=args.yes,
                         record=True,
                         additional_constraints=[pathlib.Path("cl_trials/gd_baseline/none")])


if __name__ == '__main__':
    main()
