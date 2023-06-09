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
    test_res = ttest_ind(df.loc[df['method_name'] == 'AllDataNoMDE']['success'],
              df.loc[df['method_name'] == 'FOCUS (ours)']['success'])
    print(f"p-value: {test_res.pvalue:.4f}")

    _, ax = boxplot(df, outdir, hue, 'normalized_model_error', "Model Error", figsize=(12, 4), palette=palette,
                    save=False)
    ax.set_xlabel(None)
    plt.savefig(outdir / 'real_model_error.png', dpi=200)

    ax = barplot_with_values(df, 'any_solved', hue, outdir, figsize=(12, 4), title="Plan to Goal Found",
                             palette=palette)
    ax.set_xlabel(None)
    plt.savefig(outdir / 'real_plan_to_goal_found.png', dpi=200)

    ax = barplot_with_values(df, 'success', hue, outdir, figsize=(12, 4), palette=palette,
                             title='Success on Real-World Rope Manipulation')
    ax.set_ylabel("Success Rate")
    plt.savefig(outdir / 'real_success.png', dpi=200)

    barplot_with_values(df, 'success_given_solved', hue, outdir, figsize=(12, 4), palette=palette,
                        title="Success (given plan to goal found)")
    plt.savefig(outdir / 'real_success_given_solved.png', dpi=200)

    if not args.no_plot:
        plt.show(block=True)


def barplot_with_values(df, y, hue, outdir, figsize, palette, title=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(ax=ax, data=df, x=hue, y=y, palette=palette, linewidth=5, errorbar=('ci', 95), **kwargs)
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2 - 0.1
        _y = p.get_y() + p.get_height() + 0.05
        value = '{:.2f}'.format(p.get_height())
        ax.text(_x, _y, value, ha="center")
    ax.set_ylim(0, 1.0)
    if title is None:
        ax.set_title(f"{y}")
    else:
        ax.set_title(title)
    ax.set_xlabel(None)
    return ax


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
