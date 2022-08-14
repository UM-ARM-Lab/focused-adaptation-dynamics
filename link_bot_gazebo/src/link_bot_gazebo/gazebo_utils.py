import contextlib
import os
import subprocess

import psutil

import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse, GetModelStateRequest


def launch_gazebo(world, stdout_filename, env=None):
    stdout_file = stdout_filename.open("w")
    sim_cmd = ["roslaunch", "link_bot_gazebo", "val.launch", "gui:=false", f"world:={world}"]
    popen_result = subprocess.Popen(sim_cmd, stdout=stdout_file, stderr=stdout_file, env=env)
    roslaunch_process = psutil.Process(popen_result.pid)
    return roslaunch_process


def kill_gazebo(roslaunch_process):
    for proc in roslaunch_process.children(recursive=True):
        proc.kill()
    roslaunch_process.kill()


def get_gazebo_kinect_pose(model_name="kinect2"):
    get_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    res: GetModelStateResponse = get_srv(GetModelStateRequest(model_name=model_name))
    return res.pose


def get_gazebo_processes():
    gazebo_uri_key = "GAZEBO_MASTER_URI"
    this_process_gazebo_uri = os.environ.get(gazebo_uri_key, None)

    disks = subprocess.run(["pgrep", "gzserver"], capture_output=True, text=True)
    output = disks.stdout
    processes = []
    for pid in output.split("\n"):
        try:
            pid = int(pid)
            try:
                proc = psutil.Process(pid)
                proc_environ = proc.environ()
                proc_gz_uri = proc_environ.get(gazebo_uri_key, None)
                if this_process_gazebo_uri is not None and proc_gz_uri != this_process_gazebo_uri:
                    continue  # Necessary when running multiple gzserver instances
                processes.append(proc)
            except psutil.NoSuchProcess:
                pass
        except ValueError:
            pass
    return processes


def statuses():
    gazebo_processes = get_gazebo_processes()
    statuses = [p.status() in ['stopped'] for p in gazebo_processes]
    return statuses


def is_suspended():
    gazebo_processes = get_gazebo_processes()
    suspended = any([p.status() in ['stopped'] for p in gazebo_processes])
    return suspended


def suspend():
    gazebo_processes = get_gazebo_processes()
    [p.suspend() for p in gazebo_processes]
    return gazebo_processes


def resume():
    gazebo_processes = get_gazebo_processes()
    [p.resume() for p in gazebo_processes]
    return gazebo_processes


@contextlib.contextmanager
def gazebo_suspended():
    initial_is_suspended = is_suspended()
    suspend()
    yield
    if not initial_is_suspended:
        resume()
