#!/usr/bin/env python
import argparse
import logging
import pathlib
import time

import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities import ros_init
from arc_utilities.filesystem_utils import mkdir_and_ask
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from mde.make_mde_dataset import make_mde_dataset


@ros_init.with_ros("make_mde_dataset")
def main():
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('checkpoint', type=str, help='dynamics model checkpoint')
    parser.add_argument('out_name', type=str, help='output dataset name')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--yes', '-y', action='store_true')

    args = parser.parse_args()

    root = pathlib.Path("mde_datasets")
    outdir = root / f"mde_{args.out_name}"
    success = mkdir_and_ask(outdir, parents=True, yes=args.yes)
    if not success:
        print(Fore.RED + "Aborting" + Fore.RESET)
        return

    rospy.loginfo(Fore.GREEN + f"Writing MDE dataset to {outdir}")
    dataset_dir = fetch_udnn_dataset(args.dataset_dir)
    make_mde_dataset(dataset_dir=dataset_dir,
                     checkpoint=args.checkpoint,
                     outdir=outdir,
                     step=args.step)


if __name__ == '__main__':
    main()
