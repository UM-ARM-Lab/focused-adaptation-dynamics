#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.modify_dynamics_dataset import modify_dynamics_dataset
from link_bot_pycommon.args import my_formatter
from moonshine.moonshine_utils import numpify, add_batch, remove_batch


def main():
    colorama.init(autoreset=True)
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("filter_dataset")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+filtered')

    def _process_example(dataset: DynamicsDataset, example: Dict):
        example = numpify(example)
        rope_points = example['link_bot'].reshape([dataset.sequence_length, -1, 3])
        min_z_in_sequence = np.amin(np.amin(rope_points, axis=0), axis=0)[2]
        if min_z_in_sequence < 0.59:
            dataset.scenario.plot_environment_rviz(example)
            example_t = remove_batch(dataset.index_time(add_batch(example), 0))
            dataset.scenario.plot_state_rviz(example_t, label='')
            return
        else:
            yield example

    modify_dynamics_dataset(args.dataset_dir, outdir, _process_example)


if __name__ == '__main__':
    main()
