from typing import Callable

import rospy
from arm_robots.robot_utils import make_follow_joint_trajectory_goal, get_ordered_tolerance_list, \
    interpolate_joint_trajectory_points, is_waypoint_reached, waypoint_error
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def follow_trajectory(trajectory: JointTrajectory, get_joint_positions: Callable,
                      command_and_simulate: Callable):
    """

    Args:
        trajectory:
        get_joint_positions: A function with signature:
            def get_joint_positions(joint_names: Optional[List[str]] = None) -> np.ndarray
        command_and_simulate:
            def command_and_simulate(desired_point, trajectory_joint_names, initial_joint_positions) -> None



    Returns:

    """
    traj_goal = make_follow_joint_trajectory_goal(trajectory)
    initial_joint_positions = get_joint_positions()

    # Interpolate the trajectory to a fine resolution
    # if you set max_step_size to be large and position tolerance to be small, then things will be jerky
    if len(trajectory.points) == 0:
        rospy.loginfo("Ignoring empty trajectory")
        return True

    # construct a list of the tolerances in order of the joint names
    trajectory_joint_names = trajectory.joint_names
    tolerance = get_ordered_tolerance_list(trajectory_joint_names, traj_goal.path_tolerance)
    goal_tolerance = get_ordered_tolerance_list(trajectory_joint_names, traj_goal.goal_tolerance, is_goal=True)
    interpolated_points = interpolate_joint_trajectory_points(trajectory.points, max_step_size=0.01)

    if len(interpolated_points) == 0:
        rospy.loginfo("Trajectory was empty after interpolation")
        return True

    trajectory_point_idx = 0
    t0 = rospy.Time.now()
    while True:
        # tiny sleep lets the listeners process messages better, results in smoother following
        rospy.sleep(1e-3)
        dt = rospy.Time.now() - t0

        # get feedback
        actual_joint_positions = get_joint_positions(trajectory_joint_names)

        actual_point = JointTrajectoryPoint(positions=actual_joint_positions, time_from_start=dt)
        while trajectory_point_idx < len(interpolated_points) - 1 and \
                is_waypoint_reached(actual_point, interpolated_points[trajectory_point_idx], tolerance):
            trajectory_point_idx += 1

        desired_point = interpolated_points[trajectory_point_idx]

        if trajectory_point_idx >= len(interpolated_points) - 1 and \
                is_waypoint_reached(actual_point, desired_point, goal_tolerance):
            return True

        command_and_simulate(desired_point, trajectory_joint_names,
                             initial_joint_positions)

        error = waypoint_error(actual_point, desired_point)
        if desired_point.time_from_start.to_sec() > 0 and dt > desired_point.time_from_start * 2.0:
            if trajectory_point_idx == len(interpolated_points) - 1:
                print(f"timeout. expected t={desired_point.time_from_start.to_sec()} but t={dt.to_sec()}." \
                      + f" error to waypoint is {error}, goal tolerance is {goal_tolerance}")
            else:
                print(f"timeout. expected t={desired_point.time_from_start.to_sec()} but t={dt.to_sec()}." \
                      + f" error to waypoint is {error}, tolerance is {tolerance}")
            return True