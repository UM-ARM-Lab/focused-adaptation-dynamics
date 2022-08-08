import numpy as np

from arc_utilities import ros_init
from arm_robots.hdt_michigan import Val
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotState
from tf.transformations import quaternion_from_euler


@ros_init.with_ros("basic_motion")
def main():
    val = Val(raise_on_failure=True)
    val.set_execute(False)
    # val.connect()

    start_state = RobotState()
    start_state.joint_state.name = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel",
                                    "joint56", "joint57", "joint41", "joint42", "joint43", "joint44", "joint45",
                                    "joint46", "joint47", "leftgripper", "leftgripper2", "joint1", "joint2", "joint3",
                                    "joint4", "joint5", "joint6", "joint7", "rightgripper", "rightgripper2"]
    start_state.joint_state.position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    pose = Pose()
    pose.position.x = 0.8
    pose.position.y = -0.2
    pose.position.z = 0.4
    q = quaternion_from_euler(0, np.pi, -np.pi / 4)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    val.plan_to_pose('right_side', 'right_tool', pose, start_state=start_state)

    print("done")

    # val.disconnect()


if __name__ == "__main__":
    main()
