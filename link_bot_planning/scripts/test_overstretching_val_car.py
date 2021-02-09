#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import hjson
import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities import ros_init
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from peter_msgs.srv import GetOverstretching, GetOverstretchingResponse, GetOverstretchingRequest


def print_state(state):
    print(state['left_gripper'], state['right_gripper'])


@ros_init.with_ros("test_overstretching")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(precision=3, suppress=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('params', type=pathlib.Path)
    parser.add_argument("test_scene", type=pathlib.Path)

    args = parser.parse_args()

    bagfile_name = args.test_scene
    rospy.loginfo(Fore.GREEN + f"Restoring scene {bagfile_name}")
    with args.params.open("r") as planner_params_file:
        planner_params = hjson.load(planner_params_file)

    service_provider = GazeboServices()
    scenario = get_scenario(planner_params['scenario'])
    scenario.on_before_get_state_or_execute_action()

    scenario.restore_from_bag(service_provider, planner_params, bagfile_name)
    scenario.grasp_rope_endpoints()

    srv = rospy.ServiceProxy("rope_3d/rope_overstretched", GetOverstretching)

    # try to overstretch and record what's happening
    for i in range(100):
        req: GetOverstretchingResponse = srv(GetOverstretchingRequest())
        print("Before Executing: ", req.overstretched, req.magnitude)
        state = scenario.get_state()

        action = {
            'left_gripper_position':  state['left_gripper'] + np.array([0.0, -0.1, -0.1]) * 0.1,
            'right_gripper_position': state['right_gripper'] + np.array([0.02, -0.03, -0.1]) * 0.1,
        }
        scenario.execute_action(action)

        rospy.sleep(1)

        req: GetOverstretchingResponse = srv(GetOverstretchingRequest())
        print("After Executing: ", req.overstretched, req.magnitude)


if __name__ == '__main__':
    main()
