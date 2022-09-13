import argparse
import pathlib

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


def ridge_plot(df, x: str, y: str):
    n = int(df['Epoch'].max())
    vert_scale = 10
    plt.figure()
    ax = plt.gca()
    for i in range(n + 1):
        df_i = df.loc[df[x] == i]
        weight_values = df_i[y].values
        weight_values.sort()
        bins = np.linspace(0, 1, 100)
        counts = hist_counts(bins, weight_values)
        normalized_counts = counts / sum(counts)

        color = cm.winter(i / n)
        ax.fill_between(bins, normalized_counts * vert_scale + i, y2=i, alpha=0.7, color=color)
        ax.axhline(y=i, linewidth=2, linestyle="-", c=color)

    ax.set_yticks(range(0, n + 1, 1))
    ax.set_yticklabels(range(1, n + 2, 1))
    ax.set_ylabel("Epoch")


def hist_counts(bins, weight_values):
    bin_width = bins[1] - bins[0]
    bins_lower = bins - bin_width / 2
    bins_upper = bins + bin_width / 2
    counts = []
    for lower, upper in zip(bins_lower, bins_upper):
        count = np.logical_and(lower < weight_values, weight_values < upper).sum()
        counts.append(count)
    counts = np.array(counts)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('mode', type=str)

    args = parser.parse_args()

    # plt.style.use('slides')

    dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.dataset_dir), mode=args.mode)

    data = []

    columns = ['Epoch', 'Weight']

    steps_per_epoch = 15
    n = 20

    for epoch_idx in range(0, n, 1):
        global_step = steps_per_epoch * epoch_idx
        model_i = load_udnn_model_wrapper(f'model-{args.checkpoint}:v{epoch_idx}')
        for inputs in dataset:
            inputs_batch = torchify(add_batch(inputs))
            outputs_batch = model_i(inputs_batch)
            low_error_mask = numpify(remove_batch(model_i.low_error_mask(inputs_batch, outputs_batch, global_step)))

            if 'time_mask' in inputs:
                n_transitions = int(sum(inputs['time_mask']) - 1)
            else:
                n_transitions = int(inputs['time_idx'].shape[0])

            for t in range(1, n_transitions):
                weight = low_error_mask[t]
                data.append([epoch_idx, weight])

    df = DataFrame(data, columns=columns)

    last_df = df.loc[df['Epoch'] == (n - 1)]
    last_weights = last_df['Weight'].values
    last_weights_near_1 = (last_weights > 0.99).sum()
    print(f'{last_weights_near_1 / last_weights.size:.0%} of transitions have weight 1')

    ridge_plot(df, x='Epoch', y='Weight')
    plt.savefig(f"results/weight_distribution-{args.checkpoint}-{args.dataset_dir.name}.png", dpi=200)


if __name__ == '__main__':
    main()
