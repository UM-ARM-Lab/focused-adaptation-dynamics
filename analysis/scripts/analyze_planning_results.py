#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.analyze_results import planning_results
from analysis.results_figures import boxplot
from arc_utilities import ros_init
from link_bot_pycommon.string_utils import shorten
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


def analyze_planning_results(args):
    outdir, df = planning_results(args.results_dirs, args.regenerate)

    def _shorten(c):
        return shorten(c.split('/')[0])[:16]

    df['x_name'] = df['classifier_name'].map(_shorten)

    # print(df[['any_solved']].to_string(index=False))
    # print(df[['success']].to_string(index=False))

    successes = (df['success'] == 1).sum()
    total = df['success'].count()
    print(f"{successes}/{total} = {successes / total}")

    hue = 'method_name'

    palette = {
        'FOCUS (ours)': '#0072B2',
        'AllDataNoMDE': '#009E73',
        'AllData':      '#D55E00',
    }

    summary_statistics = df.groupby("method_name").agg({"success": ["mean", "std"]})
    print(summary_statistics)
    from scipy.stats import ttest_ind
    ttest_ind(df.loc[df['method_name'] == 'AllDataNoMDE']['success'],
              df.loc[df['method_name'] == 'FOCUS (ours)']['success'])

    _, ax = boxplot(df, outdir, hue, 'task_error', "Task Error", figsize=(12, 8), palette=palette)
    ax.axhline(y=0.045, linestyle='--')
    plt.savefig(outdir / f'task_error.png')

    boxplot(df, outdir, hue, 'normalized_model_error', "Model Error", figsize=(12, 8), palette=palette)

    barplot_with_values(df, 'any_solved', hue, outdir, figsize=(12, 8), title="Any Plans Found?", palette=palette)

    barplot_with_values(df, 'success', hue, outdir, figsize=(12, 8), palette=palette)

    _, ax = boxplot(df, outdir, hue, 'task_error_given_solved', "Task Error (given solved)", figsize=(12, 8),
                    palette=palette)
    ax.axhline(y=0.045, linestyle='--')
    plt.savefig(outdir / f'task_error_given_solved.png')

    barplot_with_values(df, 'success_given_solved', hue, outdir, figsize=(12, 8), palette=palette)

    boxplot(df, outdir, hue, 'num_failed_actions', "Num Failed Actions", figsize=(12, 8), palette=palette)

    if not args.no_plot:
        plt.show(block=True)


def barplot_with_values(df, y, hue, outdir, figsize, palette, title=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(ax=ax, data=df, x=hue, y=y, palette=palette, linewidth=5, errorbar=None, **kwargs)
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height() + 0.02
        value = '{:.2f}'.format(p.get_height())
        ax.text(_x, _y, value, ha="center")
    ax.set_ylim(0, 1.0)
    if title is None:
        ax.set_title(f"{y}")
    else:
        ax.set_title(title)
    plt.savefig(outdir / f'{y}.png')


@ros_init.with_ros("analyse_planning_results")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--style', default='slides')

    args = parser.parse_args()

    plt.style.use(args.style)

    analyze_planning_results(args)


if __name__ == '__main__':
    main()
