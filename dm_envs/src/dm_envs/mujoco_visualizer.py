import warnings

from dm_control.mjcf import Physics

import rospy
from ros_numpy import msgify
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray


class MujocoVisualizer:

    def __init__(self):
        self.geoms_markers_pub = rospy.Publisher("mj_geoms", MarkerArray, queue_size=10)
        self.camera_img_pub = rospy.Publisher("mj_camera", Image, queue_size=10)

    def viz(self, physics: Physics):
        geoms_marker_msg = MarkerArray()
        self.geoms_markers_pub.publish(geoms_marker_msg)

        img = physics.render()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_msg = msgify(Image, img, encoding='rgb8')
        self.camera_img_pub.publish(img_msg)