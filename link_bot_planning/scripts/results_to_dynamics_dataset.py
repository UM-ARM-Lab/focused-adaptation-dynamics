#!/usr/bin/env python
import argparse
import pathlib

from arc_utilities import ros_init
from link_bot_data.wandb_datasets import wandb_save_dataset
from link_bot_planning.results_to_dynamics_dataset import ResultsToDynamicsDataset


@ros_init.with_ros("results_to_dataset")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("outname", help='output directory name')
    parser.add_argument("--traj-length", type=int, default=10, help='if supplied, only generate trajs of this length')
    parser.add_argument("--visualize", action='store_true')

    args = parser.parse_args()

    r = ResultsToDynamicsDataset(results_dir=args.results_dir,
                                 outname=args.outname,
                                 traj_length=args.traj_length,
                                 visualize=args.visualize)
    r.run()

    wandb_save_dataset(r.outdir, project='udnn')


if __name__ == '__main__':
    main()
