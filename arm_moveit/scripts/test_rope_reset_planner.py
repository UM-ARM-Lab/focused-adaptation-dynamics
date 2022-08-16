#!/usr/bin/env python
from copy import deepcopy

import numpy as np
from ompl import geometric as og
from pyrope_reset_planner import RopeResetPlanner

import ros_numpy
import rospy
from arc_utilities import ros_init
from arm_robots.hdt_michigan import Val
from geometry_msgs.msg import Pose, Quaternion
from link_bot_pycommon.get_scenario import get_scenario
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from tf.transformations import quaternion_from_euler
from trajectory_msgs.msg import JointTrajectoryPoint


# Author: Mark Moll


def isStateValid(state):
    # FIXME: should do collision checking with the environment and itself
    return True


def ompl_to_trajectory_msg(space, traj_msg: RobotTrajectory, ompl_sln: og.PathGeometric):
    # modifies traj_msg
    traj_msg.joint_trajectory.header.stamp = rospy.Time.now()
    traj_msg.joint_trajectory.joint_names = ['joint56', 'joint57', 'joint41', 'joint42', 'joint43', 'joint44',
                                             'joint45', 'joint46', 'joint47', 'joint1', 'joint2', 'joint3', 'joint4',
                                             'joint5', 'joint6', 'joint7']

    for state in ompl_sln.getStates():
        point = JointTrajectoryPoint()
        for i in range(space.getDimension()):
            point.positions.append(state[i])
        traj_msg.joint_trajectory.points.append(point)


@ros_init.with_ros("ompl_demo")
def main():
    left_preferred_tool_orientation = quaternion_from_euler(-1.779, -1.043, -2.0)
    right_preferred_tool_orientation = quaternion_from_euler(np.pi, -1.408, 0.9)

    scenario = get_scenario('dual_arm_rope_sim_val_with_robot_feasibility_checking', {'rope_name': 'rope_3d_alt'})
    scenario.on_before_get_state_or_execute_action()
    val: Val = scenario.robot.called
    val.plan_to_joint_config('both_arms', 'home')
    # scenario.grasp_rope_endpoints()

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

    rope_reset_planner = RopeResetPlanner()

    traj_msg = rope_reset_planner.plan_to_reset(left_tool_grasp_pose, right_tool_grasp_pose, 10)
    display_msg = DisplayTrajectory()
    display_msg.trajectory.append(traj_msg)
    pub = rospy.Publisher("ompl_plan", DisplayTrajectory, queue_size=10)
    for _ in range(10):
        rospy.sleep(0.1)
        pub.publish(display_msg)
    val.raise_on_failure = False
    execution_result = val.follow_arms_joint_trajectory(traj_msg.joint_trajectory)

    rope_reset_planner.plan_to_start()


if __name__ == "__main__":
    main()
