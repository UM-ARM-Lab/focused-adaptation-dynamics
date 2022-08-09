from copy import deepcopy

import numpy as np

import arm_robots.robot
import moveit_commander.exception
import ros_numpy
from arc_utilities import ros_init
from arm_robots.hdt_michigan import Val
from geometry_msgs.msg import Pose, Point, Quaternion
from link_bot_pycommon.get_scenario import get_scenario
from moveit_msgs.msg import RobotState
from tf.transformations import quaternion_from_euler


def test_rope_reset():
    scenario = get_scenario('dual_arm_rope_sim_val_with_robot_feasibility_checking', {'rope_name': 'rope_3d_alt'})
    scenario.on_before_get_state_or_execute_action()

    val = scenario.robot

    scenario.grasp_rope_endpoints()

    rng = np.random.RandomState(1)

    reset_config = {
        'joint1':  1.8450517654418945,
        'joint2':  0.00019175345369149,
        'joint3':  0.06807247549295425,
        'joint4':  -1.0124582052230835,
        'joint41': 1.7581875324249268,
        'joint42': 0.0,
        'joint43': 0.1562790721654892,
        'joint44': -0.9140887260437012,
        'joint45': 1.601524829864502,
        'joint46': -1.2170592546463013,
        'joint47': -0.016490796580910683,
        'joint5':  -1.5196460485458374,
        'joint56': 0.0,
        'joint57': 0.0,
        'joint6':  1.3154287338256836,
        'joint7':  -0.01706605777144432,
    }

    grasp_rope_config = {
        'joint1':  3.04,
        'joint2':  -0.006711370777338743,
        'joint3':  0.21399685740470886,
        'joint4':  -1.1934734582901,
        'joint41': 1.37,
        'joint42': 0.39,
        'joint43': 1.418016791343689,
        'joint44': -3.1,
        'joint45': 2.644280195236206,
        'joint46': 0.10929946601390839,
        'joint47': 0.013422741554677486,
        'joint5':  -2.3,
        'joint56': -0.3288571834564209,
        'joint57': 0.19,
        'joint6':  1.19,
        'joint7':  0.08609730005264282,
    }

    while True:
        plan_to_random_config(grasp_rope_config.keys(), val, rng)

        scenario.detach_rope_from_gripper('left_gripper')

        # move to reset position
        timeout = 30
        val.plan_to_joint_config("both_arms", grasp_rope_config, timeout=timeout)

        tool_names = ['left_tool', 'right_tool']

        old_tool_orientations = deepcopy(val.stored_tool_orientations)
        val.store_current_tool_orientations(tool_names)
        current_right_pos = ros_numpy.numpify(val.get_link_pose('right_tool').position)

        # move up
        left_up = ros_numpy.numpify(val.get_link_pose('left_tool').position) + np.array([0, 0, 0.1])
        val.follow_jacobian_to_position('both_arms', tool_names, [[left_up], [current_right_pos]])

        scenario.grasp_rope_endpoints()

        # go to the start config
        val.plan_to_joint_config("both_arms", reset_config, timeout=timeout)

        # restore old tool orientations
        if old_tool_orientations is not None:
            val.store_tool_orientations(old_tool_orientations)


def plan_to_random_config(joint_names, val, rng):
    while True:
        random_config = dict(zip(joint_names, rng.uniform(-1, 1, len(joint_names))))
        try:
            val.plan_to_joint_config("both_arms", random_config)
            break
        except (moveit_commander.exception.MoveItCommanderException, arm_robots.robot.RobotPlanningError):
            pass


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
