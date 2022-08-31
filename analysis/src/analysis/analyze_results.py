import multiprocessing
import pathlib
import pickle
import re
from typing import List, Dict

import pandas as pd
from hjson import HjsonDecodeError
from tqdm import tqdm

from analysis.results_metrics import metrics_funcs
from analysis.results_metrics import metrics_names
from analysis.results_utils import get_all_results_subdirs
from link_bot_pycommon.get_scenario import get_scenario_cached
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson

column_names = [
                   'data_filename',
                   'results_folder_name',
                   'method_name',
                   'seed',
                   'ift_iteration',
                   'trial_idx',
                   'uuid',
               ] + metrics_names


def make_row_worker(args):
    data_filename, metadata, scenario_params = args
    scenario = get_scenario_cached(metadata['planner_params']['scenario'], params=scenario_params)
    datum = load_gzipped_pickle(data_filename)
    row = make_row(datum, data_filename, metadata, scenario)
    return row


def make_dataframe_worker(args):
    d, regenerate = args
    data_filenames = list(d.glob("*_metrics.pkl.gz"))
    df_filename = d / 'df.pkl'
    metadata_filename = d / 'metadata.hjson'
    try:
        metadata = load_hjson(metadata_filename)
    except HjsonDecodeError:
        print(f"Bad file {metadata_filename.as_posix()}")
    if not df_filename.exists() or regenerate:
        scenario_params = dict(metadata['planner_params'].get('scenario_params', {'rope_name': 'rope_3d'}))

        # scenario = get_scenario_cached(metadata['planner_params']['scenario'], params=scenario_params)
        # NOTE: hard-coded here because using the "real val" scenario wasn't working for some reason
        scenario = get_scenario_cached('dual_arm_rope_sim_val_with_robot_feasibility_checking', params=scenario_params)
        data = []
        for data_filename in data_filenames:
            datum = load_gzipped_pickle(data_filename)
            row = make_row(datum, data_filename, metadata, scenario)
            data.append(row)

        df_i = pd.DataFrame(data)
        with df_filename.open("wb") as f:
            pickle.dump(df_i, f)
    else:
        with df_filename.open("rb") as f:
            df_i = pickle.load(f)

    return df_i


def load_planning_results(results_dirs: List[pathlib.Path], regenerate: bool = False):
    args = [(results_dir, regenerate) for results_dir in results_dirs]
    with multiprocessing.get_context("spawn").Pool() as p:
        dfs = list(tqdm(p.imap_unordered(make_dataframe_worker, args), total=len(results_dirs)))

    df = pd.concat(dfs, ignore_index=True)
    df.columns = column_names
    return df


def make_row(datum: Dict, data_filename: pathlib.Path, metadata: Dict, scenario: ScenarioWithVisualization):
    metrics_values = [metric_func(data_filename, scenario, metadata, datum) for metric_func in metrics_funcs]
    trial_idx = datum['trial_idx']
    try:
        seed_guess = datum['steps'][0]['planning_query'].seed - 100000 * trial_idx
    except (KeyError, IndexError):
        seed_guess = 0
    seed = datum.get('seed', seed_guess)

    results_folder_name = guess_results_folder_name(data_filename)

    ift_iteration = metadata.get('ift_iteration', None)
    try:
        m = re.search(r'.*iter([0-9]+)', results_folder_name)
        if m:
            ift_iteration = int(m.group(1))
    except ValueError:
        pass
    try:
        m = re.search(r'.*iteration_([0-9]+)', results_folder_name)
        if m:
            ift_iteration = int(m.group(1))
    except ValueError:
        pass

    row = [
        data_filename.as_posix(),
        results_folder_name,
        metadata['planner_params']['method_name'],
        seed,
        ift_iteration,
        trial_idx,
        datum['uuid'],
    ]
    row += metrics_values
    return row


def guess_results_folder_name(data_filename):
    results_folders = data_filename.parts[:-1]
    results_folder_name = pathlib.Path(*results_folders[-2:]).as_posix()
    return results_folder_name


def planning_results(results_dirs, regenerate=False):
    # The default for where we write results
    outdir = results_dirs[0]

    print(f"Writing analysis to {outdir}")

    results_dirs = get_all_results_subdirs(results_dirs, regenerate=regenerate)
    df = load_planning_results(results_dirs, regenerate=regenerate)
    df.to_csv("/media/shared/analysis/tmp_results.csv")

    return outdir, df


def try_split_model_name(checkpoint):
    if ':' in checkpoint:
        if 'p:' in checkpoint:
            print("Wrong format for checkpoint + label name")
            return checkpoint, checkpoint
        checkpoint, label_name = checkpoint.split(":")
        return checkpoint, label_name
    return checkpoint, checkpoint
