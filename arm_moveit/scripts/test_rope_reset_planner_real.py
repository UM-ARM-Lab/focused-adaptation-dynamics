#!/usr/bin/env python
from copy import deepcopy

import ros_numpy
from arc_utilities import ros_init
from arm_robots.hdt_michigan import Val
from geometry_msgs.msg import Pose, Quaternion
from link_bot_pycommon.get_scenario import get_scenario


@ros_init.with_ros("test_pyrope_reset_planner")
def main():
    import ompl.util as ou
    ou.setLogLevel(ou.LOG_DEBUG)

    scenario = get_scenario('real_val_with_robot_feasibility_checking', {'rope_name': 'rope_3d_alt'})
    scenario.on_before_get_state_or_execute_action()  # TODO: set preferred tool orientations in real_val_* constructors
    val: Val = scenario.robot.called

    left_tool_post_planning_pose = Pose()
    left_tool_post_planning_pose.orientation = ros_numpy.msgify(Quaternion, scenario.left_preferred_tool_orientation)
    left_tool_post_planning_pose.position.x = -0.2
    left_tool_post_planning_pose.position.y = 0.6
    left_tool_post_planning_pose.position.z = 0.34
    right_tool_post_planning_pose = deepcopy(left_tool_post_planning_pose)
    right_tool_post_planning_pose.orientation = ros_numpy.msgify(Quaternion, scenario.right_preferred_tool_orientation)
    right_tool_post_planning_pose.position.x = 0.2

    both_tools = ['left_tool', 'right_tool']
    for i in range(100):
        val.plan_to_poses('both_arms', both_tools, [left_tool_post_planning_pose, right_tool_post_planning_pose])
        scenario.reset_to_start({}, {})


if __name__ == "__main__":
    main()
