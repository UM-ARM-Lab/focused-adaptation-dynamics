#!/usr/bin/env python
import argparse
import pathlib
import re

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
        if name not in iterations_completed_map:
            iterations_completed_map[name] = {}
        iterations_completed_map[name][seed] = (full_run_name, completed_iters)

    post_learning_evaluations_map = {}
    planning_eval_root = pathlib.Path("/media/shared/planning_results")
    for planning_eval_dir in planning_eval_root.iterdir():
        if not planning_eval_dir.is_dir():
            continue
        for full_run_name, _ in full_run_names_map.items():
            m = re.search(f"{full_run_name}_iter(\d+)-{planner_config_name}", planning_eval_dir.name)
            if m:
                eval_at_iter = int(m.group(1))
                n_evals = len(list(planning_eval_dir.glob("*_metrics.pkl.gz")))
                if full_run_name not in post_learning_evaluations_map:
                    post_learning_evaluations_map[full_run_name] = {}
                post_learning_evaluations_map[full_run_name][eval_at_iter] = n_evals
    post_learning_evaluations_map = dict(sorted(post_learning_evaluations_map.items()))

    print_status(post_learning_evaluations_map)
    print('-' * 64)
    print_things_to_run(iterations_completed_map, post_learning_evaluations_map, planner_config_name)


def print_status(post_learning_evaluations_map):
    import tabulate
    table = []
    headers = ['run'] + [f'iter{i}' for i in range(10)]
    for run_name, evals in post_learning_evaluations_map.items():
        row = [0] * 11
        row[0] = f"{run_name:30s}"
        for i in range(10):
            if i in evals:
                n_evals = evals[i]
            else:
                n_evals = 0
            if n_evals == 0:
                color = Style.DIM
            elif n_evals >= 20:
                color = Style.DIM + Fore.GREEN
            else:
                color = Fore.YELLOW
            row[i + 1] = color + str(n_evals) + Style.RESET_ALL
        table.append(row)
    print(tabulate.tabulate(table, headers=headers, tablefmt='simple'))


def print_things_to_run(iterations_completed_map, post_learning_evaluations_map, planner_config_name):
    full_cmds = []
    for name, runs_for_name in iterations_completed_map.items():
        for seed, (full_run_name, _) in runs_for_name.items():
            post_learning_eval_done = True
            for iter_idx in range(10):
                if iter_idx not in post_learning_evaluations_map[full_run_name]:
                    post_learning_eval_done = False
                else:
                    n_evals = post_learning_evaluations_map[full_run_name][iter_idx]
                    if n_evals < 20:
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
