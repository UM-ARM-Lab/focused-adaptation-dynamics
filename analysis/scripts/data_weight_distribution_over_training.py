#!/usr/bin/env python
import argparse
import pathlib
import pickle

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


def ridge_plot(df, x: str, y: str, bins=50, vert_scale=10, lims=None):
    n = int(df['Epoch'].max())
    plt.figure()
    ax = plt.gca()
    for i in range(n + 1):
        df_i = df.loc[df[x] == i]
        values = df_i[y].values
        values.sort()
        counts, bin_edges = np.histogram(values, bins=bins, range=lims)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        normalized_counts = counts / sum(counts)

        color = cm.winter(i / n)
        ax.fill_between(bin_centers, normalized_counts * vert_scale + i, y2=i, alpha=0.7, color=color)
        ax.axhline(y=i, linewidth=1, linestyle="-", c=color)

    ax.set_yticks(range(0, n + 1, 1))
    ax.set_yticklabels(range(1, n + 2, 1))
    ax.set_ylabel("Epoch")
    ax.set_xlabel("Weight")
    if lims is not None:
        ax.set_xlim(lims)
    return ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('mode', type=str)
    parser.add_argument('--regenerate', action='store_true')

    args = parser.parse_args()

    plt.style.use('slides')

    dataset = TorchDynamicsDataset(fetch_udnn_dataset(args.dataset_dir), mode=args.mode)

    data = []

    steps_per_epoch = 15
    n = 20

    if args.regenerate:
        for epoch_idx in range(0, n, 1):
            global_step = steps_per_epoch * epoch_idx
            model_i = load_udnn_model_wrapper(f'model-{args.checkpoint}:v{epoch_idx}')
            for inputs in dataset:
                inputs_batch = torchify(add_batch(inputs))
                outputs_batch = model_i(inputs_batch)
                error = model_i.scenario.classifier_distance_torch(inputs_batch, outputs_batch)
                low_error_mask = numpify(remove_batch(model_i.low_error_mask(inputs_batch, outputs_batch, global_step)))

                if 'time_mask' in inputs:
                    n_transitions = int(sum(inputs['time_mask']) - 1)
                else:
                    n_transitions = int(inputs['time_idx'].shape[0])

                for t in range(1, n_transitions):
                    weight = low_error_mask[t]
                    e_t = float(error[0, t])
                    data.append([epoch_idx, weight, e_t, t, inputs['example_idx']])

        df = DataFrame(data, columns=['Epoch', 'Weight', 'Error', 't', 'example_idx'])

        with open("results/data_weight_distribution.pkl", 'wb') as f:
            pickle.dump(df, f)
    else:
        with open("results/data_weight_distribution.pkl", 'rb') as f:
            df = pickle.load(f)

    fractions_below_gamma = []
    # gamma = model_i.hparams['mask_threshold']
    gamma = 0.08
    for i in range(n):
        first_df = df.loc[df['Epoch'] == i]
        first_errors = first_df['Error'].values
        first_low_errors = (first_errors < gamma).sum()
        fraction_below_gamma = first_low_errors / first_errors.size
        print(f'{fraction_below_gamma:.0%} of transitions have low error at epoch 0')
        fractions_below_gamma.append(fraction_below_gamma)

    ax = ridge_plot(df, x='Epoch', y='Error', bins=50, vert_scale=20, lims=[0, 1])
    ax.axvline(gamma, linestyle='--', c='k', label=r'$\gamma$')
    ax.legend()
    plt.savefig(f"results/error_distribution-{args.checkpoint}-{args.dataset_dir.name}.png", dpi=200)

    plt.figure()
    plt.plot(fractions_below_gamma)
    plt.xlabel("Epoch")
    plt.ylabel("% Of Training Data")
    plt.ylim(0, 1)
    plt.title(r"Data With Prediction Error < $\gamma$")
    plt.savefig(f"results/low_error-{args.checkpoint}-{args.dataset_dir.name}.png", dpi=200)

    last_df = df.loc[df['Epoch'] == (n - 1)]
    last_weights = last_df['Weight'].values
    last_weights_near_1 = (last_weights > 0.5).sum()
    print(f'{last_weights_near_1 / last_weights.size:.0%} of transitions have weight >0.5')

    ridge_plot(df, x='Epoch', y='Weight', lims=[0, 1], vert_scale=6)
    plt.savefig(f"results/weight_distribution-{args.checkpoint}-{args.dataset_dir.name}.png", dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
