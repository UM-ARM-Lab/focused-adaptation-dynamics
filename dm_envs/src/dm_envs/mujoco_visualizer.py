import warnings

import numpy as np
from dm_control.mjcf import Physics
from mujoco import mju_str2Type, mju_mat2Quat, mjtGeom, mj_id2name, mju_mulQuat, mju_negQuat

import rospy
from ros_numpy import msgify
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker


class MujocoVisualizer:

    def __init__(self):
        self.geoms_markers_pub = rospy.Publisher("mj_geoms", MarkerArray, queue_size=10)
        self.camera_img_pub = rospy.Publisher("mj_camera", Image, queue_size=10)

    def viz(self, physics: Physics):
        from time import perf_counter
        t0 = perf_counter()
        img = physics.render(camera_id='mycamera')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_msg = msgify(Image, img, encoding='rgb8')
        self.camera_img_pub.publish(img_msg)

        geoms_marker_msg = MarkerArray()

        for geom_id in range(physics.model.ngeom):
            geom_name = mj_id2name(physics.model.ptr, mju_str2Type('geom'), geom_id)

            geom_bodyid = physics.model.geom_bodyid[geom_id]
            body_name = mj_id2name(physics.model.ptr, mju_str2Type('body'), geom_bodyid)

            if geom_name != 'val/box':
                continue

            geom_marker_msg = Marker()
            geom_marker_msg.action = Marker.ADD
            geom_marker_msg.header.frame_id = 'world'
            geom_marker_msg.ns = f'{body_name}-{geom_name}'
            geom_marker_msg.id = geom_id

            geom_type = physics.model.geom_type[geom_id]
            body_pos = physics.data.xpos[geom_bodyid]
            body_xmat = physics.data.xmat[geom_bodyid]
            body_xquat = np.zeros(4)
            mju_mat2Quat(body_xquat, body_xmat)
            geom_pos = physics.data.geom_xpos[geom_id]
            geom_xmat = physics.data.geom_xmat[geom_id]
            geom_xquat = np.zeros(4)
            mju_mat2Quat(geom_xquat, geom_xmat)
            mesh_geom_pos = physics.model.geom_pos[geom_id]
            mesh_goem_quat = physics.model.geom_quat[geom_id]
            geom_size = physics.model.geom_size[geom_id]
            geom_rgba = physics.model.geom_rgba[geom_id]
            geom_meshid = physics.model.geom_dataid[geom_id]

            geom_marker_msg.pose.position.x = geom_pos[0]
            geom_marker_msg.pose.position.y = geom_pos[1]
            geom_marker_msg.pose.position.z = geom_pos[2]
            geom_marker_msg.pose.orientation.w = geom_xquat[0]
            geom_marker_msg.pose.orientation.x = geom_xquat[1]
            geom_marker_msg.pose.orientation.y = geom_xquat[2]
            geom_marker_msg.pose.orientation.z = geom_xquat[3]
            geom_marker_msg.color.r = geom_rgba[0]
            geom_marker_msg.color.g = geom_rgba[1]
            geom_marker_msg.color.g = geom_rgba[2]
            geom_marker_msg.color.a = geom_rgba[3]

            if geom_type == mjtGeom.mjGEOM_BOX:
                geom_marker_msg.type = Marker.CUBE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[1] * 2
                geom_marker_msg.scale.z = geom_size[2] * 2
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                geom_marker_msg.type = Marker.CYLINDER
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[1] * 2
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                geom_marker_msg.type = Marker.CYLINDER  # FIXME: not accurate, should use 2 spheres and a cylinder?
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[1] * 2
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                geom_marker_msg.type = Marker.SPHERE
                geom_marker_msg.scale.x = geom_size[0] * 2
                geom_marker_msg.scale.y = geom_size[0] * 2
                geom_marker_msg.scale.z = geom_size[0] * 2
            elif geom_type == mjtGeom.mjGEOM_MESH:
                geom_marker_msg.type = Marker.MESH_RESOURCE
                mesh_name = mj_id2name(physics.model.ptr, mju_str2Type('mesh'), geom_meshid)
                mesh_name = mesh_name.split("/")[1]  # skip the model prefix, e.g. val/my_mesh
                geom_marker_msg.mesh_resource = f"package://dm_envs/meshes/{mesh_name}.stl"

                # final_geom_pos = geom_pos - mesh_geom_pos
                # neg_mesh_geom_quat = np.zeros(4)
                # mju_negQuat(neg_mesh_geom_quat, mesh_goem_quat)
                # final_geom_xquat = np.zeros(4)
                # mju_mulQuat(final_geom_xquat, geom_xquat, neg_mesh_geom_quat)
                final_geom_pos = body_pos
                final_geom_xquat = body_xquat

                geom_marker_msg.pose.position.x = final_geom_pos[0]
                geom_marker_msg.pose.position.y = final_geom_pos[1]
                geom_marker_msg.pose.position.z = final_geom_pos[2]
                geom_marker_msg.pose.orientation.w = final_geom_xquat[0]
                geom_marker_msg.pose.orientation.x = final_geom_xquat[1]
                geom_marker_msg.pose.orientation.y = final_geom_xquat[2]
                geom_marker_msg.pose.orientation.z = final_geom_xquat[3]

                geom_marker_msg.scale.x = 1
                geom_marker_msg.scale.y = 1
                geom_marker_msg.scale.z = 1
            else:
                rospy.loginfo_once(f"Unsupported geom type {geom_type}")
                continue

            geoms_marker_msg.markers.append(geom_marker_msg)

        self.geoms_markers_pub.publish(geoms_marker_msg)
        # print(f"viz took {perf_counter() - t0:0.3f}")
