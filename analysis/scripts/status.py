#!/usr/bin/env python
import argparse
import pathlib
import re
from datetime import datetime, timedelta
from time import strftime

from colorama import Fore, Style

from moonshine.filepath_tools import load_hjson


def get_run_dirs(name):
    root = pathlib.Path("/media/shared/online_adaptation")
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        if name in subdir.name:
            yield subdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()

    planner_config_name = 'soe'

    # - check how the planning evaluations have been run for each run_dir
    iterations_completed_map = {}
    full_run_names_map = {}
    for run_dir in get_run_dirs(args.name):
        full_run_name = run_dir.name
        name, seed = get_name_and_seed(run_dir)
        full_run_names_map[full_run_name] = name
        completed_iters = 0
        log = load_hjson(run_dir / 'logfile.hjson')
        for i in range(10):
            k = f'iter{i}'
            if k in log:
                log_i = log[k]
                if 'no_mde' in name:
                    done_key = 'dynamics_run_id'
                else:
                    done_key = 'mde_run_id'
                if done_key in log_i:
                    completed_iters += 1
                else:
                    break

        last_updated = None
        for hjson_path in run_dir.rglob("*.hjson"):
            time = datetime.fromtimestamp(hjson_path.stat().st_mtime)
            if last_updated is None or time > last_updated:
                last_updated = time

        if name not in iterations_completed_map:
            iterations_completed_map[name] = {}
        iterations_completed_map[name][seed] = (full_run_name, completed_iters, last_updated)

    post_learning_evaluations_map = {}
    planning_eval_root = pathlib.Path("/media/shared/planning_results")
    for planning_eval_dir in planning_eval_root.iterdir():
        if not planning_eval_dir.is_dir():
            continue
        for full_run_name, run_name in full_run_names_map.items():
            if re.match(f"{full_run_name}_iter9-{planner_config_name}", planning_eval_dir.name):
                n_evals = len(list(planning_eval_dir.glob("*_metrics.pkl.gz")))
                if run_name not in post_learning_evaluations_map:
                    post_learning_evaluations_map[run_name] = 0
                post_learning_evaluations_map[run_name] += n_evals

    print_status(iterations_completed_map, post_learning_evaluations_map)
    print('')
    print('-' * 64)
    print('')
    print_things_to_run(iterations_completed_map, post_learning_evaluations_map, planner_config_name)


def print_status(iterations_completed_map, post_learning_evaluations_map):
    n_seeds = 5
    n_total_evals = n_seeds * 64
    print(Style.BRIGHT + "Online Learning:" + Style.RESET_ALL)
    for name, runs_for_name in iterations_completed_map.items():
        print(f"{name}")
        for seed, (_, completed_iters, last_updated) in runs_for_name.items():
            if completed_iters == 10:
                color = Style.DIM
                dt_str = ""
            else:
                color = ''
                dt = datetime.now() - last_updated
                if dt > timedelta(hours=1):
                    dt_color = Fore.RED
                else:
                    dt_color = Fore.GREEN
                hours, remainder = divmod(dt.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                dt_str = dt_color + f"Last Updated {hours}hr {minutes}m {seconds}s" + Fore.RESET
            print(color + f"\tSeed {seed} Completed: {completed_iters}/10 {dt_str}" + Style.RESET_ALL)
    print('')

    print(Style.BRIGHT + "Post-Learning Evaluations:" + Style.RESET_ALL)
    for name, n_evals in post_learning_evaluations_map.items():
        print(f"{name:20s} {n_evals:4d}/{n_total_evals}")


def print_things_to_run(iterations_completed_map, post_learning_evaluations_map, planner_config_name):
    print("Things to run:")
    for name, runs_for_name in iterations_completed_map.items():
        for seed in range(10):
            if seed not in runs_for_name:
                print(f"./scripts/online_rope_sim.py {name}-{seed}")
    print('')
    for name, runs_for_name in iterations_completed_map.items():
        for seed, (full_run_name, completed_iters, _) in runs_for_name.items():
            # for name, n_evals in post_learning_evaluations_map.items():
            online_is_done = (completed_iters == 10)
            planning_eval_root = pathlib.Path("/media/shared/planning_results")
            post_learning_eval_started = (planning_eval_root / f"{full_run_name}_iter9-{planner_config_name}").exists()
            if online_is_done and not post_learning_eval_started:
                planner_config_path = f"planner_configs/val_car/{planner_config_name}.hjson"
                online_learning_dir = f"/media/shared/online_adaptation/{full_run_name}"
                print(f"./scripts/evaluate_online_iter.py {planner_config_path} {online_learning_dir} 9")


def get_name_and_seed(run_dir):
    if '-' in run_dir.name:
        name_without_seed, seed = run_dir.name.split('-')
        seed = int(seed)
    else:
        name_without_seed = run_dir.name
        seed = 0
    if 'all_data' not in name_without_seed and "adaptation" not in name_without_seed:
        name_without_seed += '_adaptation'
    return name_without_seed, seed


if __name__ == '__main__':
    main()
