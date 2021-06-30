import pathlib
import pickle
import logging
from typing import List

import pandas as pd
from tqdm import tqdm

from analysis.figspec import DEFAULT_AXES_NAMES, FigSpec, TableSpec
from analysis.results_metrics import metrics_funcs
from analysis.results_metrics import metrics_names
# noinspection PyUnresolvedReferences
from analysis.results_tables import *
from link_bot_pycommon.get_scenario import get_scenario_cached
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson

logger = logging.getLogger(__file__)

column_names = [
                   'method_name',
                   'seed',
                   'ift_iteration',
                   'trial_idx',
                   'uuid',
               ] + metrics_names


def load_fig_specs(analysis_params, figures_config: pathlib.Path):
    figures_config = load_hjson(figures_config)
    figspecs = []
    for fig_config in figures_config:
        figure_type = eval(fig_config.pop('type'))
        reductions = fig_config.pop('reductions')
        axes_names = DEFAULT_AXES_NAMES

        fig = figure_type(analysis_params, **fig_config)

        figspec = FigSpec(fig=fig, reductions=reductions, axes_names=axes_names)
        figspecs.append(figspec)
    return figspecs


def load_table_specs(tables_config: pathlib.Path, table_format: str):
    tables_conf = load_hjson(tables_config)
    tablespecs = []
    for table_conf in tables_conf:
        table_type = eval(table_conf.pop('type'))
        reductions = table_conf.pop('reductions')
        axes_names = DEFAULT_AXES_NAMES

        table = table_type(table_format=table_format, **table_conf)

        tablespec = TableSpec(table=table, reductions=reductions, axes_names=axes_names)
        tablespecs.append(tablespec)
    return tablespecs


def reduce_planning_metrics(reductions: List[List], metrics: pd.DataFrame):
    reduced_metrics = []
    for reduction in reductions:
        metric_i = metrics.copy()
        for reduction_step in reduction:
            group_by, metric, agg = reduction_step
            assert metric is not None
            if group_by is None or len(group_by) == 0:
                metric_i = metric_i.agg({metric: agg})
            elif group_by is not None and agg is not None:
                metric_i = metric_i.groupby(group_by).agg({metric: agg})
            elif group_by is not None and agg is None:
                metric_i.set_index(group_by, inplace=True)
                metric_i = metric_i[metric]
            else:
                raise NotImplementedError()
        reduced_metrics.append(metric_i)

    reduced_metrics = pd.concat(reduced_metrics, axis=1)
    return reduced_metrics


def load_planning_results(results_dirs: List[pathlib.Path], regenerate: bool):
    dfs = []
    for d in tqdm(results_dirs, desc='results dirs'):
        data_filenames = list(d.glob("*_metrics.pkl.gz"))
        df_filename = d / 'df.pkl'
        metadata_filename = d / 'metadata.hjson'
        metadata = load_hjson(metadata_filename)
        if not df_filename.exists() or regenerate:
            scenario = get_scenario_cached(metadata['planner_params']['scenario'])
            data = []
            for data_filename in tqdm(data_filenames, desc='results files'):
                datum = load_gzipped_pickle(data_filename)
                try:
                    row = make_row(datum, metadata, scenario)
                except:
                    logger.error(data_filename)
                    continue

                data.append(row)
            df_i = pd.DataFrame(data)
            with df_filename.open("wb") as f:
                pickle.dump(df_i, f)
        else:
            with df_filename.open("rb") as f:
                df_i = pickle.load(f)
        dfs.append(df_i)

    df = pd.concat(dfs)
    df.columns = column_names
    return df


def make_row(datum, metadata, scenario):
    metrics_values = [metric_func(scenario, metadata, datum) for metric_func in metrics_funcs]
    trial_idx = datum['trial_idx']
    try:
        seed_guess = datum['steps'][0]['planning_query'].seed - 100000 * trial_idx
    except KeyError:
        seed_guess = 0
    seed = datum.get('seed', seed_guess)
    row = [
        metadata['planner_params']['method_name'],
        seed,
        metadata.get('ift_iteration', 0),
        trial_idx,
        datum['uuid'],
    ]
    row += metrics_values
    return row
