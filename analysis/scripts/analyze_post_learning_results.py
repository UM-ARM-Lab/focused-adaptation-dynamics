#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import boxplot, barplot
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def metrics_main(args):
    outdir, df = planning_results(args.results_dirs, args.regenerate)

    # Filter the rows to keep only the trails with the same planning conditions
    # df = df_where(df, 'max_extensions_param', 5_000)
    # df = df_where(df, 'max_attempts', 3)
    # df = df_where(df, 'max_planning_attempts', 3)

    # if the results_folder_name contains the key, the set method_name to be the value
    method_name_map = {
        # order here matters
        'all_data_no_mde': 'all_data_no_mde',
        'all_data':        'all_data',
        '':                'adaptation',
    }

    method_names = []
    for results_name in df['results_folder_name']:
        for substr, method_name in method_name_map.items():
            if substr in results_name:
                method_names.append(method_name)
                break

    df['method_name'] = method_names
    df = df.sort_values("method_name")

    fig, ax = barplot(df, outdir, 'method_name', 'success', 'Success', ci=90)
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(outdir / "success.png")
    fig, ax = barplot(df, outdir, 'method_name', 'success_given_solved', 'Success (given solved)', ci=90)
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(outdir / "success_given_solved.png")
    fig, ax = boxplot(df, outdir, 'method_name', 'task_error', 'Task Error')
    fig, ax = boxplot(df, outdir, 'method_name', 'combined_error', 'Combined Error')
    fig, ax = boxplot(df, outdir, 'method_name', 'normalized_model_error', 'Model Error')
    fig, ax = boxplot(df, outdir, 'method_name', 'any_solved', 'Plans Found?')
    plt.show()


@ros_init.with_ros("analyse_online_results")
def main():
    pd.options.display.max_rows = 999

    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--style', default='slides')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    plt.style.use(args.style)
    plt.rcParams['figure.figsize'] = (20, 10)
    sns.set(rc={'figure.figsize': (7, 4)})

    metrics_main(args)


if __name__ == '__main__':
    import numpy as np

    np.seterr(all='raise')  # DEBUGGING
    main()
