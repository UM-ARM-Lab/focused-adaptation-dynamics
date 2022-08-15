from copy import deepcopy
from time import sleep

import numpy as np

import ros_numpy
from arc_utilities import ros_init
from arm_robots.hdt_michigan import Val
from arm_robots.robot import RobotPlanningError, MoveitEnabledRobot
from arm_robots.robot_utils import PlanningResult, PlanningAndExecutionResult
from geometry_msgs.msg import Pose, Point, Quaternion
from link_bot_pycommon.get_scenario import get_scenario
from moveit_msgs.msg import RobotState, MoveItErrorCodes, Constraints, OrientationConstraint
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler


def plan_to_reset(self: MoveitEnabledRobot, group_name, ee_links, target_poses, right_tool_orientation):
    for ee_link, target_pose_i in zip(ee_links, target_poses):
        self.display_goal_pose(target_pose_i, ee_link)

    nearest_ik_solution, _ = self.nearby_ik(ee_links, group_name, target_poses)

    group_joint_names = self.get_joint_names(group_name)
    target_joint_config = {}
    for n in group_joint_names:
        i = nearest_ik_solution.joint_state.name.index(n)
        p = nearest_ik_solution.joint_state.position[i]
        target_joint_config[n] = p

    print(list(target_joint_config.values()))

    # return self.plan_to_joint_config(group_name, target_joint_config)
    move_group = self.get_move_group_commander(group_name)

    move_group.set_joint_value_target(target_joint_config)
    right_tool_constraint = Constraints()
    right_tool_orientation_constraint = OrientationConstraint()
    right_tool_orientation_constraint.orientation = ros_numpy.msgify(Quaternion, right_tool_orientation)
    right_tool_orientation_constraint.header.frame_id = 'right_tool'
    right_tool_orientation_constraint.link_name = 'right_tool'
    right_tool_orientation_constraint.weight = 1
    right_tool_orientation_constraint.parameterization = OrientationConstraint.XYZ_EULER_ANGLES
    right_tool_orientation_constraint.absolute_x_axis_tolerance = 0.1
    right_tool_orientation_constraint.absolute_y_axis_tolerance = 0.1
    right_tool_orientation_constraint.absolute_z_axis_tolerance = 0.1
    right_tool_constraint.orientation_constraints.append(right_tool_orientation_constraint)
    move_group.set_path_constraints(right_tool_constraint)

    if self.display_goals:
        robot_state = RobotState(joint_state=JointState(name=list(target_joint_config.keys()),
                                                        position=list(target_joint_config.values())))
        self.display_robot_state(robot_state, label='joint_config_goal')

    planning_result = PlanningResult(move_group.plan())
    if planning_result.planning_error_code.val == MoveItErrorCodes.INVALID_MOTION_PLAN:
        print("Invalid plan, trying to replan.")
        planning_result = PlanningResult(move_group.plan())

    if self.raise_on_failure and not planning_result.success:
        raise RobotPlanningError(f"Plan to joint config failed {planning_result.planning_error_code}")
    execution_result = self.follow_arms_joint_trajectory(planning_result.plan.joint_trajectory)
    return PlanningAndExecutionResult(planning_result, execution_result)


def test_rope_reset():
    scenario = get_scenario('dual_arm_rope_sim_val_with_robot_feasibility_checking', {'rope_name': 'rope_3d_alt'})

    scenario.on_before_get_state_or_execute_action()

    val: Val = scenario.robot.called
    # val.raise_on_failure = False

    scenario.grasp_rope_endpoints()

    rng = np.random.RandomState(1)

    left_preferred_tool_orientation = quaternion_from_euler(-1.779, -1.043, -2.0)
    right_preferred_tool_orientation = quaternion_from_euler(np.pi, -1.408, 0.9)
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

    both_tools = ['left_tool', 'right_tool']

    for j in range(100):
        while True:
            # plan to a start config with the right tool orientations
            # left_tool_post_planning_pose.position.x += rng.uniform(-0.01, 0.01)
            # left_tool_post_planning_pose.position.y += rng.uniform(-0.01, 0.01)
            # left_tool_post_planning_pose.position.z += rng.uniform(-0.01, 0.01)
            # right_tool_post_planning_pose.position.x += rng.uniform(-0.01, 0.01)
            # right_tool_post_planning_pose.position.y += rng.uniform(-0.01, 0.01)
            # right_tool_post_planning_pose.position.z += rng.uniform(-0.01, 0.01)
            res = val.plan_to_poses('both_arms', both_tools,
                                    [left_tool_post_planning_pose, right_tool_post_planning_pose])

            print(val.get_joint_positions(val.get_joint_names('both_arms')))

            scenario.detach_rope_from_gripper('left_gripper')

            # move to reset position
            plan_to_reset(val, 'both_arms', both_tools, [left_tool_grasp_pose, right_tool_grasp_pose],
                          right_preferred_tool_orientation)

            # move up
            sleep(1)
            right_grasp_position_np = ros_numpy.numpify(right_tool_grasp_pose.position)
            left_up = ros_numpy.numpify(val.get_link_pose('left_tool').position) + np.array([0, 0, 0.08])
            val.store_current_tool_orientations(both_tools)
            val.follow_jacobian_to_position('both_arms', both_tools, [[left_up], [right_grasp_position_np]])

            scenario.grasp_rope_endpoints()

            # go to the start config
            val.plan_to_poses("both_arms", both_tools, [left_start_pose, right_start_pose])

            val.store_tool_orientations({
                'left_tool':  left_preferred_tool_orientation,
                'right_tool': right_preferred_tool_orientation,
            })

            state = scenario.get_state()
            rope_points = state['rope'].reshape([-1, 3])
            midpoint = rope_points[12]
            rope_is_in_box = np.linalg.norm(midpoint - np.array([0.8, 0, 0.6])) < 0.1
            if not scenario.is_rope_overstretched() and rope_is_in_box:
                break

        print("done", j)


@ros_init.with_ros("test_dumb_planning")
def main():
    # test_plan_to_pose(val)
    test_rope_reset()

    # val.disconnect()


def test_plan_to_pose():
    val = Val(raise_on_failure=True)
    val.set_execute(False)

    timeout = 3
    start_state = make_start_state()

    nominal_position = np.array([0.8, 0, 0.4])
    nominal_orientation_euler = np.array([0, np.pi, -np.pi / 4])
    pose = Pose()
    rng = np.random.RandomState(0)
    while True:
        start_state.joint_state.position = rng.uniform(-0.1, 0.1, 24)
        position = nominal_position + rng.uniform(-0.01, 0.01, 3)
        orientation_euler = nominal_orientation_euler + rng.uniform(-.01, .01, 3)
        orientation = quaternion_from_euler(*orientation_euler)
        pose.position = ros_numpy.msgify(Point, position)
        pose.orientation = ros_numpy.msgify(Quaternion, orientation)

        val.plan_to_pose('right_arm', 'right_tool', pose, start_state=start_state, timeout=timeout)
        # val.plan_to_pose('left_arm', 'left_tool', pose, start_state=start_state, timeout=timeout)


def make_start_state():
    start_state = RobotState()
    start_state.joint_state.name = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel",
                                    "joint56", "joint57", "joint41", "joint42", "joint43", "joint44", "joint45",
                                    "joint46", "joint47", "leftgripper", "leftgripper2", "joint1", "joint2", "joint3",
                                    "joint4", "joint5", "joint6", "joint7", "rightgripper", "rightgripper2"]
    start_state.joint_state.position = np.zeros(len(start_state.joint_state.name))
    return start_state


if __name__ == "__main__":
    main()
