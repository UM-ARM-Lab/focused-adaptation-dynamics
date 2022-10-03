#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from moonshine.torch_and_tf_utils import add_batch
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', default='paper')

    config = {
        'rope':  {
            'dataset_dir':           pathlib.Path('known_good_4'),
            'mode':                  'test',
            'all_data_checkpoint':   'validating_all_data-1kjen',
            'adaptation_checkpoint': 'validating_adaptation-3s298',
        },
        'water': {
            'dataset_dir':           pathlib.Path('known_good_all_dynamics_dataset_all'),
            'mode':                  'all',
            'all_data_checkpoint':   'fine_tune_all_data_biggerdataset2-6zara',
            'adaptation_checkpoint': 'fine_tune_adaptation_biggerdataset2-97c3n',
        },

    }

    args = parser.parse_args()

    data = []

    for domain_name, domain_config in config.items():
        dataset = TorchDynamicsDataset(fetch_udnn_dataset(domain_config['dataset_dir']), mode=domain_config['mode'])
        checkpoints = {
            'AllData':      domain_config['all_data_checkpoint'],
            'FOCUS (ours)': domain_config['adaptation_checkpoint'],
        }
        for (name, checkpoint) in checkpoints.items():
            model = load_udnn_model_wrapper(checkpoint)

            for inputs in dataset:
                inputs_batch = torchify(add_batch(inputs))
                outputs_batch = model(inputs_batch)
                error = model.scenario.classifier_distance_torch(inputs_batch, outputs_batch)

                if 'time_mask' in inputs:
                    n_transitions = int(sum(inputs['time_mask']) - 1)
                else:
                    n_transitions = int(inputs['time_idx'].shape[0])

                for t in range(1, n_transitions):
                    e_t = float(error[0, t])
                    data.append([domain_name, name, checkpoint, e_t])
    df = pd.DataFrame(data, columns=['domain_name', 'name', 'checkpoint', 'error'])

    plt.style.use(args.style)
    plt.rcParams['figure.figsize'] = (12, 5)

    palette = {
        'FOCUS (ours)': '#0072B2',
        'AllData':      '#D55E00',
        'AllDataNoMde': '#009E73',
    }

    sns.boxplot(data=df, x='name', y='error', palette=palette, showfliers=False)
    plt.ylabel("Prediction Error")
    plt.title(r"Error on an approximate $\mathcal{D}_{ST}$")
    plt.savefig('results/known_good_model_error.png', dpi=300)
    plt.show()
