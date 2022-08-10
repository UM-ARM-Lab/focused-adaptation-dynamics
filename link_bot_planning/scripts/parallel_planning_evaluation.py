#!/usr/bin/env python
import argparse
import math
import os
import pathlib
import subprocess
import time

from link_bot_planning.eval_online_utils import evaluate_online_iter_outdir
from link_bot_pycommon.args import int_set_arg


def main(args):
    planning_processes = []
    num_parallel_data_collection_threads = 1
    trial_idxs = args.trials
    trials_per_thread = math.ceil(len(trial_idxs) / num_parallel_data_collection_threads)

    port_num = 11320

    for process_idx in range(num_parallel_data_collection_threads):
        outdir = evaluate_online_iter_outdir(args.planner_params, args.online_dir, args.iter)
        outdir.mkdir(exist_ok=True)

        stdout_filename = outdir / f'{process_idx}.stdout'
        stdout_file = stdout_filename.open("w")
        stderr_filename = outdir / f'{process_idx}.stderr'
        stderr_file = stderr_filename.open("w")
        sim_stdout_filename = outdir / f'{process_idx}_sim.stdout'
        sim_stdout_file = sim_stdout_filename.open("w")
        sim_stderr_filename = outdir / f'{process_idx}_sim.stderr'
        sim_stderr_file = sim_stderr_filename.open("w")

        env = os.environ.copy()
        env["GAZEBO_MASTER_URI"] = f"http://localhost:{port_num}"
        sim_cmd = ["roslaunch", "link_bot_gazebo", "val.launch", "gui:=false", f"world:={args.world}"]
        env["ROS_MASTER_URI"] = f"http://localhost:{port_num + 1}"
        print("starting sim", process_idx)
        subprocess.Popen(sim_cmd, env=env, stdout=sim_stdout_file, stderr=sim_stderr_file)

        time.sleep(5)

        trial_start_idx = process_idx * trials_per_thread
        trial_end_idx = min(trial_start_idx + trials_per_thread - 1, len(trial_idxs) + 1)
        trials_set = f"{trial_idxs[trial_start_idx]}-{trial_idxs[trial_end_idx]}"
        planning_cmd = ["python", "scripts/evaluate_online_iter.py", args.planner_params, args.online_dir,
                        str(args.iter), f"--trials={trials_set}", "--on-exception=raise", '-y']
        port_num += 2
        print("starting planning", process_idx)
        planning_process = subprocess.Popen(planning_cmd, env=env, stdout=stdout_file, stderr=stderr_file)

        time.sleep(10)

        planning_processes.append(planning_process)

    for planning_process in planning_processes:
        planning_process.wait()

    print("Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params', type=pathlib.Path, default="planner_configs/val_car/mde_no_replanning.hjson")
    parser.add_argument('world', type=str)
    parser.add_argument('online_dir', type=pathlib.Path)
    parser.add_argument('iter', type=int)
    parser.add_argument("--trials", type=int_set_arg, default="1-9")
    args = parser.parse_args()
    main(args)
