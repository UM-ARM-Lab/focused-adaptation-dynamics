#!/usr/bin/env python
from copy import deepcopy
from time import sleep

import numpy as np
from pyrope_reset_planner import RopeResetPlanner, PlanningResult

import ros_numpy
import rospy
from arc_utilities import ros_init
from arm_robots.hdt_michigan import Val
from geometry_msgs.msg import Pose, Quaternion
from link_bot_pycommon.get_scenario import get_scenario
from moveit_msgs.msg import DisplayTrajectory
from tf.transformations import quaternion_from_euler


@ros_init.with_ros("test_pyrope_reset_planner")
def main():
    # ou.setLogLevel(ou.LOG_ERROR)

    left_preferred_tool_orientation = quaternion_from_euler(-1.779, -1.043, -2.0)
    right_preferred_tool_orientation = quaternion_from_euler(np.pi, -1.408, 0.9)

    scenario = get_scenario('dual_arm_rope_sim_val_with_robot_feasibility_checking', {'rope_name': 'rope_3d_alt'})
    scenario.on_before_get_state_or_execute_action()
    val: Val = scenario.robot.called
    scenario.grasp_rope_endpoints()

    val.store_tool_orientations({
        'left_tool':  left_preferred_tool_orientation,
        'right_tool': right_preferred_tool_orientation,
    })

    left_tool_post_planning_pose = Pose()
    left_tool_post_planning_pose.orientation = ros_numpy.msgify(Quaternion, left_preferred_tool_orientation)
    left_tool_post_planning_pose.position.x = 1.0
    left_tool_post_planning_pose.position.y = 0.2
    left_tool_post_planning_pose.position.z = 0.5
    right_tool_post_planning_pose = Pose()
    right_tool_post_planning_pose.orientation = ros_numpy.msgify(Quaternion, right_preferred_tool_orientation)
    right_tool_post_planning_pose.position.x = 1.0
    right_tool_post_planning_pose.position.y = -0.2
    right_tool_post_planning_pose.position.z = 0.5

    left_start_pose = Pose()
    left_start_pose.orientation = ros_numpy.msgify(Quaternion, left_preferred_tool_orientation)
    left_start_pose.position.x = 0.8
    left_start_pose.position.y = 0.2
    left_start_pose.position.z = 0.9
    right_start_pose = Pose()
    right_start_pose.orientation = ros_numpy.msgify(Quaternion, right_preferred_tool_orientation)
    right_start_pose.position.x = 0.8
    right_start_pose.position.y = -0.2
    right_start_pose.position.z = 0.9

    right_tool_grasp_pose = Pose()
    right_tool_grasp_pose.position.x = 0.7
    right_tool_grasp_pose.position.y = -0.1
    right_tool_grasp_pose.position.z = 1.1
    right_tool_grasp_pose.orientation = ros_numpy.msgify(Quaternion, right_preferred_tool_orientation)

    left_tool_grasp_pose = deepcopy(right_tool_grasp_pose)
    left_tool_grasp_pose.position.z = right_tool_grasp_pose.position.z - 0.8
    left_tool_grasp_pose.orientation = ros_numpy.msgify(Quaternion,
                                                        quaternion_from_euler(0, np.pi / 2 + 0.2, -np.pi / 2))

    scenario.tf.send_transform_from_pose_msg(left_tool_grasp_pose, "robot_root", "left_grasp_pose")

    both_tools = ['left_tool', 'right_tool']
    for i in range(100):
        while True:
            res = val.plan_to_poses('both_arms', both_tools,
                                    [left_tool_post_planning_pose, right_tool_post_planning_pose])

            scenario.detach_rope_from_gripper('left_gripper')

            rrp = RopeResetPlanner()

            plan_to_grasp(left_tool_grasp_pose, right_tool_grasp_pose, rrp, val)

            sleep(1)
            right_grasp_position_np = ros_numpy.numpify(right_tool_grasp_pose.position)
            left_up = ros_numpy.numpify(left_tool_grasp_pose.position) + np.array([0, 0, 0.08])
            val.store_current_tool_orientations(both_tools)
            val.follow_jacobian_to_position('both_arms', both_tools, [[left_up], [right_grasp_position_np]])

            scenario.grasp_rope_endpoints()

            plan_to_start(left_start_pose, right_start_pose, rrp, val)

            state = scenario.get_state()
            rope_points = state['rope'].reshape([-1, 3])
            midpoint = rope_points[12]
            rope_is_in_box = np.linalg.norm(midpoint - np.array([0.8, 0, 0.6])) < 0.1
            if not scenario.is_rope_overstretched() and rope_is_in_box:
                break
        print(f"done {i}")


def plan_to_start(left_start_pose, right_start_pose, rrp, val):
    result: PlanningResult = rrp.plan_to_start(left_start_pose, right_start_pose, 0.8, 1.2, 0.1, 30)
    if result.status != "Exact solution":
        print("BAD SOLUTION!!!")
    display_msg = DisplayTrajectory()
    display_msg.trajectory.append(result.traj)
    pub = rospy.Publisher("/test_rope_reset_planner/ompl_plan", DisplayTrajectory, queue_size=10)
    for _ in range(10):
        rospy.sleep(0.1)
        pub.publish(display_msg)
    val.raise_on_failure = False
    result.traj.joint_trajectory.header.stamp = rospy.Time.now()
    execution_result = val.follow_arms_joint_trajectory(result.traj.joint_trajectory)


def plan_to_grasp(left_tool_grasp_pose, right_tool_grasp_pose, rrp, val):
    orientation_path_tol = 0.1
    while True:
        result = rrp.plan_to_reset(left_tool_grasp_pose, right_tool_grasp_pose, orientation_path_tol, 0.1, 10)
        orientation_path_tol *= 2
        if result.status == "Exact solution":
            break
    display_msg = DisplayTrajectory()
    display_msg.trajectory.append(result.traj)
    pub = rospy.Publisher("/test_rope_reset_planner/ompl_plan", DisplayTrajectory, queue_size=10)
    for _ in range(10):
        rospy.sleep(0.1)
        pub.publish(display_msg)
    val.raise_on_failure = False
    result.traj.joint_trajectory.header.stamp = rospy.Time.now()
    execution_result = val.follow_arms_joint_trajectory(result.traj.joint_trajectory)


if __name__ == "__main__":
    main()
