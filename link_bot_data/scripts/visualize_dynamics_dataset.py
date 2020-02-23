#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_data.visualization import plot_rope_configuration, plottable_rope_configuration
from link_bot_planning.params import FullEnvParams
from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=3)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train', help='train test or val')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--redraw', action='store_true', help='redraw')

    args = parser.parse_args()

    np.random.seed(1)
    tf.random.set_random_seed(1)

    dataset = LinkBotStateSpaceDataset(args.dataset_dir)
    train_dataset = dataset.get_datasets(mode=args.mode,
                                         sequence_length=None,
                                         n_parallel_calls=1)
    if args.shuffle:
        train_dataset = train_dataset.shuffle(1024, seed=1)

    full_env_params = FullEnvParams.from_json(dataset.hparams['full_env_params'])

    half_w = full_env_params.w / 2 - 0.01
    half_h = full_env_params.h / 2 - 0.01

    def oob(point):
        if point[0] <= -half_w or point[0] >= half_w:
            return True
        if point[1] <= -half_h or point[1] >= half_h:
            return True
        return False

    # print info about shapes
    input_data, output_data = next(iter(train_dataset))
    print('Inputs:')
    for k, v in input_data.items():
        print(k, v.shape)
    print('Outputs:')
    for k, v in output_data.items():
        print(k, v.shape)

    i = 0
    all_vs = []
    for input_data, output_data in train_dataset:

        out_of_bounds = False

        tether = ('state/tether' in input_data)

        rope_configurations = input_data['state/link_bot'].numpy()
        if tether:
            tether_configurations = input_data['state/tether'].numpy()
        actions = input_data['action'].numpy()
        all_vs.extend(actions.flatten().tolist())

        if not args.no_plot:
            full_env = input_data['full_env/env'].numpy()
            full_env_extents = input_data['full_env/extent'].numpy()

            fig, ax = plt.subplots()
            arrow_width = 0.1
            arena_size = 0.5

            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_xlim([-arena_size, arena_size])
            ax.set_ylim([-arena_size, arena_size])
            ax.axis("equal")

            full_env_handle = ax.imshow(np.flipud(full_env), extent=full_env_extents)
            head_point = rope_configurations[0].reshape(-1, 2)[-1]
            arrow = plt.Arrow(head_point[0], head_point[1], actions[0, 0], actions[0, 1], width=arrow_width, zorder=4)
            patch = ax.add_patch(arrow)
            link_bot_line = plot_rope_configuration(ax, rope_configurations[0], linewidth=5, zorder=3, c='r')[0]
            if tether:
                tether_line = plot_rope_configuration(ax, tether_configurations[0], linewidth=1, zorder=2, c='w')[0]

            def update(t):
                nonlocal patch
                link_bot_config = rope_configurations[t]
                tether_config = tether_configurations[t]
                head_point = link_bot_config.reshape(-1, 2)[-1]
                action = actions[t]

                full_env_handle.set_data(np.flipud(full_env))
                full_env_handle.set_extent(full_env_extents)

                if args.redraw:
                    xs, ys = plottable_rope_configuration(link_bot_config)
                    link_bot_line.set_xdata(xs)
                    link_bot_line.set_ydata(ys)
                    if tether:
                        xs, ys = plottable_rope_configuration(tether_config)
                        tether_line.set_xdata(xs)
                        tether_line.set_ydata(ys)
                else:
                    plot_rope_configuration(ax, link_bot_config, linewidth=5, zorder=3, c='r')
                    if tether:
                        plot_rope_configuration(ax, tether_config, linewidth=1, zorder=2, c='w')
                patch.remove()
                arrow = plt.Arrow(head_point[0], head_point[1], action[0], action[1], width=arrow_width, zorder=4)
                patch = ax.add_patch(arrow)

                ax.set_title("{} {}".format(i, t))

            interval = 50
            _ = FuncAnimation(fig, update, frames=actions.shape[0], interval=interval, repeat=True)
            plt.show()

        for config in rope_configurations:
            for p in config.reshape([-1, 2]):
                out_of_bounds = out_of_bounds or oob(p)
            if out_of_bounds:
                print("Example {} is goes of bounds!".format(i))
                break

        i += 1
    print("dataset mode={}, size={}".format(args.mode, i))

    if not args.no_plot:
        plt.hist(all_vs)
        plt.show()


if __name__ == '__main__':
    main()
