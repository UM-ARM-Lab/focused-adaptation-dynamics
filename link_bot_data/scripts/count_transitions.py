#!/usr/bin/env python

import argparse
import pathlib

from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from moonshine.my_torch_dataset import MyTorchDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    for mode in ['train', 'val', 'test']:
        dataset = MyTorchDataset(fetch_udnn_dataset(args.dataset_dir), mode=mode)
        n_transitions = 0
        for e in dataset:
            if 'time_mask' in e:
                n_transitions += int(sum(e['time_mask']) - 1)
            else:
                n_transitions += int(e['time_idx'].shape[0])
        print(mode, n_transitions)


if __name__ == '__main__':
    main()
