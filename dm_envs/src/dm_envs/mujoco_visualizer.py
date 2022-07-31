import warnings

import mujoco
import numpy as np
from dm_control.mjcf import Physics
from transformations import quaternion_from_matrix

import rospy
from geometry_msgs.msg import Point
from ros_numpy import msgify
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker


class MujocoVisualizer:

    def __init__(self):
        self.geoms_markers_pub = rospy.Publisher("mj_geoms", MarkerArray, queue_size=10)
        self.camera_img_pub = rospy.Publisher("mj_camera", Image, queue_size=10)

    def viz(self, physics: Physics):
        img = physics.render(camera_id='mycamera')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_msg = msgify(Image, img, encoding='rgb8')
        self.camera_img_pub.publish(img_msg)

        geoms_marker_msg = MarkerArray()

        for geom_id in range(physics.model.ngeom):
            geom_name = mujoco.mj_id2name(physics.model.ptr, mujoco.mju_str2Type('geom'), geom_id)

            geom_bodyid = physics.model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(physics.model.ptr, mujoco.mju_str2Type('body'), geom_bodyid)

            # print('goem:', geom_name)
            # print('body:', body_name)
            geom_marker_msg = Marker()
            geom_marker_msg.action = Marker.ADD
            geom_marker_msg.header.frame_id = 'world'
            geom_marker_msg.ns = f'{body_name}-{geom_name}'
            geom_marker_msg.id = geom_id

            geom_type = physics.model.geom_type[geom_id]
            geom_pos = physics.data.geom_xpos[geom_id]
            geom_xmat = physics.data.geom_xmat[geom_id].reshape([3, 3])
            geom2world = np.eye(4)
            geom2world[:3, :3] = geom_xmat
            geom2world[:3, -1] = geom_pos
            geom_quat = quaternion_from_matrix(geom2world)
            geom_size = physics.model.geom_size[geom_id]
            geom_rgba = physics.model.geom_rgba[geom_id]
            geom_dataid = physics.model.geom_dataid[geom_id]

            geom_marker_msg.pose.position.x = geom_pos[0]
            geom_marker_msg.pose.position.y = geom_pos[1]
            geom_marker_msg.pose.position.z = geom_pos[2]
            geom_marker_msg.pose.orientation.w = geom_quat[0]
            geom_marker_msg.pose.orientation.x = geom_quat[1]
            geom_marker_msg.pose.orientation.y = geom_quat[2]
            geom_marker_msg.pose.orientation.z = geom_quat[3]
            geom_marker_msg.color.r = geom_rgba[0]
            geom_marker_msg.color.g = geom_rgba[1]
            geom_marker_msg.color.g = geom_rgba[2]
            geom_marker_msg.color.a = geom_rgba[3]

            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                geom_marker_msg.type = Marker.CUBE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[1] * 2
                geom_marker_msg.scale.z = geom_size[2] * 2
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom_marker_msg.type = Marker.CYLINDER
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[1] * 2
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                geom_marker_msg.type = Marker.CYLINDER  # FIXME: not accurate, should use 2 spheres and a cylinder?
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[1] * 2
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                geom_marker_msg.type = Marker.SPHERE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[0] * 2
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                geom_marker_msg.type = Marker.TRIANGLE_LIST

                mesh = physics.model.mesh(geom_dataid)

                for face_i in range(0, mesh.facenum[0], 1):
                    # Points here are in geom frame
                    face_adr = mesh.faceadr[0] + face_i
                    point_adrs = physics.model.mesh_face[face_adr]
                    point0_xyz = physics.model.mesh_vert[mesh.vertadr[0] + point_adrs[0]]
                    point1_xyz = physics.model.mesh_vert[mesh.vertadr[0] + point_adrs[1]]
                    point2_xyz = physics.model.mesh_vert[mesh.vertadr[0] + point_adrs[2]]

                    geom_marker_msg.points.append(msgify(Point, point0_xyz))
                    geom_marker_msg.points.append(msgify(Point, point1_xyz))
                    geom_marker_msg.points.append(msgify(Point, point2_xyz))
                geom_marker_msg.scale.x = 1
                geom_marker_msg.scale.y = 1
                geom_marker_msg.scale.z = 1

            else:
                print(f"Unsupported geom type {geom_type}")
                continue

            geoms_marker_msg.markers.append(geom_marker_msg)

        self.geoms_markers_pub.publish(geoms_marker_msg)
        print('pub!')
