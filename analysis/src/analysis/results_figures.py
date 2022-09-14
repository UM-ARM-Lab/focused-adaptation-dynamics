from typing import Optional

import seaborn as sns
from matplotlib import pyplot as plt


def try_set_violinplot_color(parts, key, color):
    if key in parts:
        parts[key].set_edgecolor(color)


def lineplot(df,
             x: str,
             metric: str,
             title: str,
             hue: Optional[str] = None,
             style: Optional[str] = None,
             figsize=None,
             scatt=False,
             palette='colorblind',
             errorbar=('ci', 90)):
    fig = plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=df.sort_values(hue),
        x=x,
        y=metric,
        hue=hue,
        style=style,
        palette=palette,
        errorbar=errorbar,
    )
    df_for_scatt = df.groupby([hue, x]).agg("mean").reset_index()
    if scatt:
        sns.scatterplot(
            data=df_for_scatt.sort_values(hue),
            x=x,
            y=metric,
            hue=hue,
            ax=ax)
    ax.plot([], [], ' ', label=f"Shaded 95% c.i.")
    ax.set_title(title)
    ax.legend()
    return fig, ax


DEFAULT_FIG_SIZE = (10, 8)


def generic_plot(plot_type, df, outdir, x: str, y: str, title: str, hue: Optional[str] = None, save: bool = True,
                 palette='colorblind', figsize=DEFAULT_FIG_SIZE, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    getattr(sns, plot_type)(
        ax=ax,
        data=df,
        x=x,
        y=y,
        palette=palette,
        hue=hue,
        linewidth=4,
        **kwargs,
    )
    ax.set_title(title)
    if save:
        plt.savefig(outdir / f'{y}.png')
    return fig, ax


def boxplot(df, outdir, x: str, y: str, title: str, hue: Optional[str] = None, save: bool = True,
            figsize=DEFAULT_FIG_SIZE,
            palette='colorblind',
            outliers=True, **kwargs):
    return generic_plot('boxplot', df, outdir, x, y, title, hue, save, palette, figsize, showfliers=outliers, **kwargs)


def violinplot(df, outdir, x: str, y: str, title: str, hue: Optional[str] = None, save: bool = True,
               figsize=DEFAULT_FIG_SIZE, **kwargs):
    return generic_plot('violinplot', df, outdir, x, y, title, hue, save, figsize, **kwargs)
