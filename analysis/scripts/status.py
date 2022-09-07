#!/usr/bin/env python
import argparse
import pathlib
import re
from datetime import timedelta, datetime
from itertools import chain

from colorama import Style, Fore

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
        logfilename = run_dir / 'logfile.hjson'
        if not logfilename.exists():
            continue
        log = load_hjson(logfilename)
        for i in range(20):
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
        for hjson_path in chain(run_dir.rglob("*.hjson"), run_dir.rglob("*.stdout")):
            time = datetime.fromtimestamp(hjson_path.stat().st_mtime)
            if last_updated is None or time > last_updated:
                last_updated = time

        if name not in iterations_completed_map:
            iterations_completed_map[name] = {}
        iterations_completed_map[name][seed] = (full_run_name, completed_iters, last_updated)

    post_learning_evaluations_map = {}
    planning_eval_root = pathlib.Path("/media/shared/planning_results")
    total_n_evals = 0
    for planning_eval_dir in planning_eval_root.iterdir():
        if not planning_eval_dir.is_dir():
            continue
        for full_run_name, _ in full_run_names_map.items():
            m = re.search(f"{full_run_name}_iter(\d+)-{planner_config_name}", planning_eval_dir.name)
            if m:
                eval_at_iter = int(m.group(1))
                n_evals = len(list(planning_eval_dir.glob("*_metrics.pkl.gz")))
                total_n_evals += n_evals
                if full_run_name not in post_learning_evaluations_map:
                    post_learning_evaluations_map[full_run_name] = {}
                post_learning_evaluations_map[full_run_name][eval_at_iter] = n_evals
    post_learning_evaluations_map = dict(sorted(post_learning_evaluations_map.items()))

    print_online_learning_status(iterations_completed_map)
    print('-' * 64)

    expected_total_n_evals = 10 * 10 * 30
    print(f"{total_n_evals}/{expected_total_n_evals} ({total_n_evals / expected_total_n_evals:.0%})")
    print_post_learning_evaluation_status(post_learning_evaluations_map)


def print_online_learning_status(iterations_completed_map):
    print(Style.BRIGHT + "Online Learning:" + Style.RESET_ALL)
    for name, runs_for_name in iterations_completed_map.items():
        print(f"{name}")
        runs_for_name_sorted = dict(sorted(runs_for_name.items()))
        for seed, (_, completed_iters, last_updated) in runs_for_name_sorted.items():
            if completed_iters == 20:
                color = Style.DIM
                dt_str = ""
            else:
                color = ''
                dt = datetime.now() - last_updated
                if dt > timedelta(minutes=20):
                    dt_color = Fore.RED
                else:
                    dt_color = Fore.GREEN
                hours, remainder = divmod(dt.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                dt_str = dt_color + f"Last Updated {hours}hr {minutes}m {seconds}s" + Fore.RESET
            print(color + f"\tSeed {seed} Completed: {completed_iters}/20 {dt_str}" + Style.RESET_ALL)


def print_post_learning_evaluation_status(post_learning_evaluations_map):
    import tabulate
    table_dict = {}
    headers = ['run'] + [f'{i}' for i in range(20)]
    for run_name, evals in post_learning_evaluations_map.items():
        name, seed = get_name_and_seed_from_name(run_name)
        if name not in table_dict:
            table_dict[name] = [0] * 21
        row = table_dict[name]
        row[0] = name
        for i in range(20):
            if i in evals:
                n_evals = evals[i]
            else:
                n_evals = 0
            row[i + 1] += n_evals
            if row[i + 1] >= 100:
                row[i + 1] = 100

    table = []
    for dict_row in table_dict.values():
        row = [dict_row[0]]
        for n_evals in dict_row[1:]:
            if n_evals >= 100:
                color = Fore.GREEN
            elif n_evals > 0:
                color = Fore.YELLOW
            else:
                color = ""
            row.append(color + str(n_evals) + Fore.RESET)
        table.append(row)
    print(tabulate.tabulate(table, headers=headers, tablefmt='simple'))


def print_things_to_run(iterations_completed_map, post_learning_evaluations_map, planner_config_name):
    full_cmds = []
    for name, runs_for_name in iterations_completed_map.items():
        for seed, (full_run_name, _, __) in runs_for_name.items():
            post_learning_eval_done = True
            for iter_idx in range(10):
                if iter_idx not in post_learning_evaluations_map[full_run_name]:
                    post_learning_eval_done = False
                else:
                    n_evals = post_learning_evaluations_map[full_run_name][iter_idx]
                    if n_evals < 10:
                        post_learning_eval_done = False
            if not post_learning_eval_done:
                planner_config_path = f"planner_configs/val_car/{planner_config_name}.hjson"
                online_learning_dir = f"/media/shared/online_adaptation/{full_run_name}"
                cmd = f"./scripts/evaluate_online_iter.py {planner_config_path} {online_learning_dir} $i -y"
                full_cmd = f"for i in {{0..9}}; do {cmd}; done;"
                full_cmds.append(full_cmd)
    full_cmds = sorted(full_cmds)

    for full_cmd in full_cmds:
        print(full_cmd)


def get_name_and_seed(run_dir):
    return get_name_and_seed_from_name(run_dir.name)


def get_name_and_seed_from_name(run_name):
    if '-' in run_name:
        name_without_seed, seed = run_name.split('-')
        seed = int(seed)
    else:
        name_without_seed = run_name
        seed = 0
    if 'all_data' not in name_without_seed and "adaptation" not in name_without_seed:
        name_without_seed += '_adaptation'
    return name_without_seed, seed


if __name__ == '__main__':
    main()
