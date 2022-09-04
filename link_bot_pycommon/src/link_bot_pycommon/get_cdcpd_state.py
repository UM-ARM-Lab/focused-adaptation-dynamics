import numpy as np

import rospy
from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper
from link_bot_pycommon.ros_pycommon import transform_points_to_robot_frame
from sensor_msgs.msg import PointCloud2


class GetCdcpdState:

    def __init__(self, tf: TF2Wrapper, root_link: str, key: str = 'rope'):
        self.tf = tf
        self.key = key
        self.cdcpd_listener = Listener("cdcpd/output", PointCloud2)
        self.root_link = root_link

    def get_state(self):
        # wait until the rope state has settled
        max_delta_m = 0.003

        def get_cdcpd_points():
            cdcpd_msg: PointCloud2 = rospy.wait_for_message("cdcpd/output", PointCloud2)
            points = transform_points_to_robot_frame(self.tf, cdcpd_msg, robot_frame_id=self.root_link)
            return points

        last_points = None
        for _ in range(20):
            points = get_cdcpd_points()
            rospy.sleep(0.5)

            if last_points is not None:
                delta_m = np.max(np.linalg.norm(points - last_points, axis=-1))
                if delta_m < max_delta_m:
                    break

            last_points = points

        cdcpd_vector = points.flatten()
        return {
            self.key: cdcpd_vector,
        }
