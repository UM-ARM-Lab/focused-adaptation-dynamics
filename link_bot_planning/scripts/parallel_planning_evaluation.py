#!/usr/bin/env python
import subprocess, os, math
import numpy as np
import pathlib
import argparse
import time
from link_bot_pycommon.args import int_set_arg


def main(args):
    gz_processes = []
    planning_processes = []
    num_parallel_data_collection_threads = 1
    trial_idxs = args.trials
    trials_per_thread = math.ceil(len(trial_idxs) / num_parallel_data_collection_threads)

    port_num = 11320

    for process_idx in range(num_parallel_data_collection_threads):
        env = os.environ.copy()
        import ipdb;
        ipdb.set_trace()
        env["GAZEBO_MASTER_URI"] = f"http://localhost:{port_num}"
        sim_cmd = ["roslaunch", "link_bot_gazebo", "val.launch", "gui:=false", "world:=car4_alt.world"]
        env["ROS_MASTER_URI"] = f"http://localhost:{port_num + 1}"
        sim_process = subprocess.Popen(sim_cmd, env=env)
        time.sleep(5)
        trial_start_idx = process_idx * trials_per_thread
        trial_end_idx = min(trial_start_idx + trials_per_thread - 1, len(trial_idxs) + 1)
        trials_set = f"{trial_idxs[trial_start_idx]}-{trial_idxs[trial_end_idx]}"
        planning_cmd = ["python", "scripts/planning_evaluation2.py", args.planner_params, "test_scenes/car4_alt",
                        args.outdir, args.dynamics, args.mde, f"--trials={trials_set}", "--on-exception=raise"]
        port_num += 2
        planning_process = subprocess.Popen(planning_cmd, env=env)
        time.sleep(10)
        planning_processes.append(planning_process)

    for planning_process in planning_processes:
        planning_process.wait()

    print("Completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner_params', type=pathlib.Path,
                        default="planner_configs/val_car/mde_no_replanning.hjson")
    parser.add_argument("--outdir", type=pathlib.Path, default="/media/shared/planning_results/gazebo_parallel6",
                        help='used in making the output directory')
    parser.add_argument('--dynamics', type=pathlib.Path, default="p:model-v7_udnn_9-1iwez:latest")
    parser.add_argument('--mde', type=pathlib.Path, default="p:model-v7_mde_9-dk87f:latest")
    parser.add_argument("--trials", type=int_set_arg, default="1-9")
    args = parser.parse_args()
    main(args)
