import os
import pathlib
import subprocess
from time import sleep
from typing import Dict, List

import more_itertools

from link_bot_gazebo import gazebo_utils


def online_parallel_planning(planner_params: Dict,
                             dynamics: str,
                             mde: str,
                             outdir: pathlib.Path,
                             test_scenes_dir: pathlib.Path,
                             method_name: str,
                             trials: List[int],
                             seed,
                             how_to_handle: str,
                             n_parallel: int,
                             world: str):
    planning_processes = []
    roslaunch_processes = []
    port_num = 42000

    for process_idx, trials_iterable in enumerate(more_itertools.divide(n_parallel, trials)):
        trials_strs = [str(trials_i) for trials_i in trials_iterable]
        trials_set = ','.join(trials_strs)
        print(process_idx, trials_set)

        stdout_filename = outdir / f'{process_idx}.stdout'
        print(f"Writing stdout/stderr to {stdout_filename}")

        env = os.environ.copy()
        env["GAZEBO_MASTER_URI"] = f"http://localhost:{port_num}"
        env["ROS_MASTER_URI"] = f"http://localhost:{port_num + 1}"
        roslaunch_process = gazebo_utils.launch_gazebo(world, stdout_filename, env=env)
        print(f"PID: {roslaunch_process.pid}")

        planning_cmd = [
            "python",
            "scripts/planning_evaluation2.py",
            planner_params,
            test_scenes_dir.as_posix(),
            outdir.as_posix(),
            dynamics,
            mde,
            f"--trials={trials_set}",
            f"--on-exception={how_to_handle}",
            f"--seed={seed}",
            f'--method-name={method_name}',
        ]
        port_num += 2
        print(f"starting planning {process_idx} for trials {trials_set}")
        eval_stdout_filename = outdir / f'{process_idx}_eval.stdout'
        eval_stdout_file = eval_stdout_filename.open("w")
        planning_process = subprocess.Popen(planning_cmd, env=env, stdout=eval_stdout_file, stderr=eval_stdout_file)
        print(f"PID: {planning_process.pid}")

        planning_processes.append(planning_process)
        roslaunch_processes.append(roslaunch_process)

    for planning_process in planning_processes:
        planning_process.wait()

    for roslaunch_process in roslaunch_processes:
        gazebo_utils.kill_gazebo(roslaunch_process)
