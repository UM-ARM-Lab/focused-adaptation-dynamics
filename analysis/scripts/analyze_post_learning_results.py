#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from analysis.analyze_results import planning_results
from analysis.results_figures import lineplot
from arc_utilities import ros_init
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def metrics_main(args):
    root = pathlib.Path("/media/shared/planning_results/")
    results_dirs = list(root.glob(args.name + "*"))
    outdir, df = planning_results(results_dirs, args.regenerate)

    # if the results_folder_name contains the key, the set method_name to be the value
    method_name_map = {
        # order here matters
        'all_data_no_mde': 'AllDataNoMDE',
        'all_data':        'AllData',
        '':                'FOCUS (ours)',
    }
    palette = {
        'FOCUS (ours)': '#0072B2',
        'AllData':      '#D55E00',
        'AllDataNoMDE': '#009E73',
    }

    method_names = []
    for results_name in df['results_folder_name']:
        for substr, method_name in method_name_map.items():
            if substr in results_name:
                method_names.append(method_name)
                break

    df['method_name'] = method_names
    df = df.sort_values("method_name")

    iter_key = 'ift_iteration'
    pi = args.pi
    errorbar = ('ci', 95)

    fig, ax = lineplot(df, x=iter_key, hue='method_name', metric='success_given_solved',
                       title='Success (given plan to goal found) ⬆', errorbar=errorbar, palette=palette)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(20))
    ax.set_xlabel("Online Learning Iteration")
    ax.set_ylabel("Success Rate")
    plt.savefig(outdir / "success_given_plan_found.png")

    fig, ax = lineplot(df, x=iter_key, hue='method_name', metric='normalized_model_error', title='Model Error ⬇',
                       errorbar=errorbar, palette=palette)
    ax.set_xlabel("Online Learning Iteration")
    ax.set_xticks(range(20))
    ax.set_ylabel("Model Error")
    plt.savefig(outdir / "model_error.png")

    fig, ax = lineplot(df, x=iter_key, hue='method_name', metric='any_solved', title='Plan to Goal Found ⬆',
                       errorbar=errorbar, palette=palette)
    ax.set_xlabel("Online Learning Iteration")
    ax.set_xticks(range(20))
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("% plan to goal found")
    plt.savefig(outdir / "plan_found.png")

    fig, ax = lineplot(df, x=iter_key, hue='method_name', metric='success', title='Success ⬆', errorbar=errorbar,
                       palette=palette)
    ax.set_xlabel("Online Learning Iteration")
    ax.set_xticks(range(20))
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(outdir / "success.png")

    if not args.no_plot:
        plt.show()


@ros_init.with_ros("analyse_online_results")
def main():
    pd.options.display.max_rows = 999

    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='results directory', type=str)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--pi', default=90, type=int)
    parser.add_argument('--style', default='paper')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    plt.style.use(args.style)
    plt.rcParams['figure.figsize'] = (14, 8)

    metrics_main(args)


if __name__ == '__main__':
    main()
