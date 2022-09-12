#!/usr/bin/env python

import argparse
import pathlib

from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    for mode in ['train', 'val', 'test']:
        dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.dataset_dir), mode=mode)
        n_transitions_in_collision = 0
        n_transitions = 0
        for e in dataset:
            # TODO: check collision with environment and robot
            if 'time_mask' in e:
                n_transitions += int(sum(e['time_mask']) - 1)
            else:
                n_transitions += int(e['time_idx'].shape[0])


if __name__ == '__main__':
    main()
