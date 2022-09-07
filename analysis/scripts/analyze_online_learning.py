#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import lineplot
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def metrics_main(args):
    outdir, df = planning_results(args.results_dirs, args.regenerate)

    # manually add the results for what iter0 would do, which are currently based on:
    iter_key = 'ift_iteration'

    fig, ax = lineplot(df, iter_key, 'normalized_model_error', 'Model Error', hue='method_name')
    ax.set_xticks(range(20))
    ax.legend()
    plt.savefig(outdir / f'normalized_model_error.png')

    fig, ax = lineplot(df, iter_key, 'success', 'Success', hue='method_name')
    ax.set_xticks(range(20))
    ax.legend()
    plt.savefig(outdir / f'success.png')

    fig, ax = lineplot(df, iter_key, 'success_given_solved', 'Success (given Plan To Goal Found)', hue='method_name')
    ax.set_xticks(range(20))
    ax.legend()
    plt.savefig(outdir / f'success_given_solved.png')

    fig, ax = lineplot(df, iter_key, 'any_solved', 'Plan to goal found?', hue='method_name')
    ax.set_xticks(range(20))
    ax.legend()
    plt.savefig(outdir / f'any_solved.png')

    fig, ax = lineplot(df, iter_key, 'task_error', 'Task Error', hue='method_name')
    ax.set_xticks(range(20))
    ax.legend()
    plt.savefig(outdir / f'task_error.png')

    if not args.no_plot:
        plt.show()


@ros_init.with_ros("analyse_ift_results")
def main():
    pd.options.display.max_rows = 999

    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--latex', action='store_true')
    parser.add_argument('--order', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--debug', action='store_true', help='will only run on a few examples to speed up debugging')
    parser.add_argument('--style', default='paper')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    plt.style.use(args.style)
    # plt.rcParams['figure.figsize'] = (20, 10)
    sns.set(rc={'figure.figsize': (6, 3)})

    metrics_main(args)


if __name__ == '__main__':
    import numpy as np

    np.seterr(all='raise')  # DEBUGGING
    main()
