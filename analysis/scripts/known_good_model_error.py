#!/usr/bin/env python
import wandb
import argparse

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', default='paper')

    args = parser.parse_args()

    run_ids = [
        'eval-gp7nu',
        'eval-dqrhe',
    ]
    plt.style.use(args.style)
    plt.rcParams['figure.figsize'] = (18, 5)
    names = [
        'AllData',
        'Adaptation (ours)',
    ]
    colors = [
        '#e3a20b',
        '#3565a1',
    ]

    api = wandb.Api()
    model_errors = []
    for run_id in run_ids:
        r = api.run(f'armlab/udnn/{run_id}')
        model_errors.append(r.summary['test_error'])

    # TODO: try a boxplot
    plt.bar(names, model_errors, color=colors)
    plt.ylabel("Model Error")
    plt.title(r"Model Error on an approximate $\mathcal{D}_{ST}$ for Rope")
    plt.savefig('results/known_good_model_error.png', dpi=300)
    plt.show()
