#!/usr/bin/env python
import argparse
import pathlib
from time import time

import numpy as np
import torch

from arc_utilities import ros_init
from link_bot_pycommon.args import run_subparsers
from mde import train_test_mde
from moonshine.magic import wandb_lightning_magic

node_name = f"train_test_mde_{int(time())}"


@ros_init.with_ros(node_name)
def main():
    def _train_main(args):
        if args.seed is None:
            args.seed = np.random.randint(0, 10000)

        train_test_mde.train_main(**vars(args))

    def _fine_tune_main(args):
        train_test_mde.fine_tune_main(**vars(args))

    def _eval_main(args):
        train_test_mde.eval_main(**vars(args))

    def _viz_main(args):
        train_test_mde.viz_main(**vars(args))

    torch.set_printoptions(linewidth=250, precision=7, sci_mode=False)
    np.set_printoptions(linewidth=250, precision=7, suppress=True)

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dir', type=pathlib.Path)
    train_parser.add_argument('params_filename', type=pathlib.Path)
    train_parser.add_argument('--nickname', '-n', type=str)
    train_parser.add_argument('--user', '-u', type=str, default='armlab')
    train_parser.add_argument('--checkpoint')
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--repeat', type=int)
    train_parser.add_argument('--skip', type=int)
    train_parser.add_argument('--epochs', type=int, default=150)
    train_parser.add_argument('--no-val', action='store_true')
    train_parser.add_argument('--is_nn_mde', type=int, default=1)
    train_parser.add_argument('--dryrun', type=int)
    train_parser.add_argument('--steps', type=int, default=-1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.set_defaults(func=_train_main)

    fine_tune_parser = subparsers.add_parser('fine_tune')
    fine_tune_parser.add_argument('dataset_dir', type=pathlib.Path, nargs='+')
    fine_tune_parser.add_argument('checkpoint', type=str)
    fine_tune_parser.add_argument('params_filename', type=pathlib.Path)
    fine_tune_parser.add_argument('--nickname', '-n', type=str)
    fine_tune_parser.add_argument('--user', '-u', type=str, default='armlab')
    fine_tune_parser.add_argument('--batch-size', type=int, default=32)
    fine_tune_parser.add_argument('--take', type=int)
    fine_tune_parser.add_argument('--repeat', type=int)
    fine_tune_parser.add_argument('--skip', type=int)
    fine_tune_parser.add_argument('--train-mode')
    fine_tune_parser.add_argument('--val-mode')
    fine_tune_parser.add_argument('--epochs', type=int, default=15)
    fine_tune_parser.add_argument('--no-val', action='store_true')
    fine_tune_parser.add_argument('--is_nn_mde', type=int, default=1)
    fine_tune_parser.add_argument('--dryrun', type=int)
    fine_tune_parser.add_argument('--steps', type=int, default=-1)
    fine_tune_parser.add_argument('--seed', type=int, default=0)
    fine_tune_parser.set_defaults(func=_fine_tune_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dir', type=pathlib.Path)
    viz_parser.add_argument('checkpoint')
    viz_parser.add_argument('--user', '-u', type=str, default='armlab')
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    viz_parser.add_argument('--skip', type=int)
    viz_parser.set_defaults(func=_viz_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dir', type=pathlib.Path)
    eval_parser.add_argument('checkpoint')
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--user', '-u', type=str, default='armlab')
    eval_parser.add_argument('--take', type=int)
    eval_parser.add_argument('--project', type=str, default="mde")
    eval_parser.add_argument('--is_nn_mde', type=int, default=1)
    eval_parser.add_argument('--beta', type=float, default=2)
    eval_parser.add_argument('--dryrun')


    eval_parser.set_defaults(func=_eval_main)

    wandb_lightning_magic()

    run_subparsers(parser)


if __name__ == '__main__':
    main()
