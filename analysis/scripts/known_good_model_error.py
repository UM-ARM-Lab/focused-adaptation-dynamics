#!/usr/bin/env python
import wandb
import argparse

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', default='paper')

    args = parser.parse_args()

    run_ids = [
        'eval-7zqi7',
        'eval-citcs',
        'eval-24m2d',
    ]
    plt.style.use(args.style)
    plt.rcParams['figure.figsize'] = (18, 5)
    names = [
        'AllData',
        'AllDataNoMDE',
        'Adaptation (ours)',
    ]
    colors = [
        '#e3a20b',
        '#599673',
        '#3565a1',
    ]

    api = wandb.Api()
    model_errors = []
    for run_id in run_ids:
        r = api.run(f'armlab/udnn/{run_id}')
        model_errors.append(r.summary['test_loss'])

    plt.bar(names, model_errors, color=colors)
    plt.ylabel("Model Error")
    plt.title("Model Error on known-free-space trajectories")
    plt.savefig('results/known_good_model_error.png', dpi=300)
    plt.show()
