#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import lineplot
from arc_utilities import ros_init
from link_bot_pycommon.metric_utils import dict_to_pvalue_table
from link_bot_pycommon.pandas_utils import rlast, df_where
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


def metrics_main(args):
    outdir, df, table_specs = planning_results(args.results_dirs, args.regenerate)

    # manually add the results for what iter0 would do, which are currently based on:
    unadapted_path = pathlib.Path(
        "media/shared/planning_results/unadapted_eval_for_online_iter0_1659632839_babbe85f3a")

    w = 5
    max_iter = 15
    x_max = max_iter + 0.01
    ci = 95
    te_max = 0.25
    nme_max = 1.2
    iter_key = 'ift_iteration'

    # z2 = df.groupby(iter_key).agg('mean').rolling(w).agg('mean')  # groupby iter_key also sorts by default
    # fig, ax = lineplot(z2, iter_key, 'success', 'Success Rate [all combined] (rolling)')
    # ax.set_ylim(-0.01, 1.01)

    # compute rolling average per run

    method_name_map = {
        '/media/shared/online_adaptation/v7':                 'Adaptation (ours)',
        '/media/shared/online_adaptation/v7_all_data':        'All Data (baseline)',
        '/media/shared/online_adaptation/v7_all_data_no_mde': 'No MDE (baseline)',
        '/media/shared/online_adaptation/v8':                 'Adaptation (ours)',
        '/media/shared/online_adaptation/v8_all_data':        'All Data (baseline)',
        '/media/shared/online_adaptation/v8_all_data_no_mde': 'No MDE (baseline)',
    }

    for i, k in enumerate(method_name_map.keys()):
        indices, = np.where(df['data_filename'].str.startswith(k))
        df.loc[indices, 'method_idx'] = i

    method_name_values = []
    for method_idx in df['method_idx'].values:
        if np.isnan(method_idx):
            method_name_values.append(np.nan)
        else:
            k = list(method_name_map.keys())[int(method_idx)]
            method_name_values.append(method_name_map[k])
    df['method_name'] = method_name_values

    agg = {
        'success':                'mean',
        'task_error':             'mean',
        'normalized_model_error': 'mean',
        'method_idx':             rlast,
        iter_key:                 rlast,
    }

    df_r = df.sort_values(iter_key).groupby('ift_uuid').rolling(w).agg(agg)
    method_name_values_r = []
    for method_idx in df_r['method_idx'].values:
        if np.isnan(method_idx):
            method_name_values_r.append(np.nan)
        else:
            k = list(method_name_map.keys())[int(method_idx)]
            method_name_values_r.append(method_name_map[k])
    df_r['method_name'] = method_name_values_r

    # pvalues_at_iter(df_r, method_name_values, 99)
    # print_values_for_ablations_table(df_r, method_name_values, 99)

    fig, ax = lineplot(df, iter_key, 'success', 'Success', hue='method_name', ci=None)
    ax.set_ylim(-0.01, 1.01)
    plt.savefig(outdir / f'success.png')

    fig, ax = lineplot(df, iter_key, 'task_error', 'Task Error', hue='method_name')
    plt.savefig(outdir / f'task_error.png')

    fig, ax = lineplot(df, iter_key, 'normalized_model_error', 'Model Error', hue='method_name')
    plt.savefig(outdir / f'normalized_model_error.png')

    if not args.no_plot:
        plt.show()


def print_values_for_ablations_table(df_r, method_name_values, ift_iter):
    final_metrics = df_where(df_r, 'ift_iteration', ift_iter)
    for n in np.unique(method_name_values):
        if n == 'nan':
            continue
        success_rates = final_metrics.loc[final_metrics['method_name'] == n]['success'].values
        print(f"{n:30s} {np.mean(success_rates):.4f} {np.std(success_rates):.4f}")


def pvalues_at_iter(df_r, method_name_values, ift_iter):
    final_metrics = df_where(df_r, 'ift_iteration', ift_iter)
    final_success_dict = {}
    for n in np.unique(method_name_values):
        if n == 'nan':
            continue
        success_rates = final_metrics.loc[final_metrics['method_name'] == n]['success'].values
        final_success_dict[n] = success_rates
    print(dict_to_pvalue_table(final_success_dict))


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
