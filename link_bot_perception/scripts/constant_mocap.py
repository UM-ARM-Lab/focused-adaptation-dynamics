#!/usr/bin/env python
import rospkg

import roslaunch
import rosnode
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from geometry_msgs.msg import Pose


def main():
    rospy.init_node('constant_mocap')

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    r = rospkg.RosPack()
    vb_path = r.get_path('lightweight_vicon_bridge')
    launch = roslaunch.parent.ROSLaunchParent(uuid, [f"{vb_path}/launch/vicon_bridge.launch"])
    launch.start()

    tf = TF2Wrapper()
    parent = 'mocap_world'
    children = [
        'mocap_kinect2_roof_kinect2_roof',
        'mocap_kinect2_tripodA_kinect2_tripodA',
        'mocap_left_hand_left_hand',
        'mocap_right_hand_right_hand',
        'mocap_val_wood_mount_val_wood_mount',
    ]

    print("Getting transforms...")
    child_poses = []
    for child in children:
        transform_stamped = tf.get_transform_msg(parent=parent, child=child)
        child_poses.append([child, transform_stamped.transform])

    rosnode.kill_nodes(['lightweight_vicon_bridge'])

    print("Publishing transforms...")
    while not rospy.is_shutdown():
        for child, transform in child_poses:
            pose = Pose()
            pose.position = transform.translation
            pose.orientation = transform.rotation
            tf.send_transform_from_pose_msg(pose, parent=parent, child=child, is_static=False)


if __name__ == '__main__':
    main()
