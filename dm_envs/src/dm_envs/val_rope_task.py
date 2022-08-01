from typing import Dict

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.utils import inverse_kinematics
from mujoco import mj_id2name, mju_str2Type, mju_mat2Quat, mjtGeom
from transformations import quaternion_from_euler

import rospy
from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from arm_robots.robot_utils import interpolate_joint_trajectory_points, get_ordered_tolerance_list, is_waypoint_reached, \
    waypoint_error, make_follow_joint_trajectory_goal
from dm_envs.base_rope_task import BaseRopeManipulation
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotState, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def make_planning_scene_msg(physics):
    msg = PlanningScene()
    msg.name = 'world'
    msg.is_diff = False

    for geom_id in range(physics.model.ngeom):
        geom_name = mj_id2name(physics.model.ptr, mju_str2Type('geom'), geom_id)

        geom_bodyid = physics.model.geom_bodyid[geom_id]
        body_name = mj_id2name(physics.model.ptr, mju_str2Type('body'), geom_bodyid)

        collision_object = CollisionObject()
        collision_object.header.frame_id = 'world'
        collision_object.operation = CollisionObject.ADD
        collision_object.id = f'{body_name}-{geom_name}'

        geom_type = physics.model.geom_type[geom_id]
        body_pos = physics.data.xpos[geom_bodyid]
        body_xmat = physics.data.xmat[geom_bodyid]
        body_xquat = np.zeros(4)
        mju_mat2Quat(body_xquat, body_xmat)
        geom_pos = physics.data.geom_xpos[geom_id]
        geom_xmat = physics.data.geom_xmat[geom_id]
        geom_xquat = np.zeros(4)
        mju_mat2Quat(geom_xquat, geom_xmat)
        geom_size = physics.model.geom_size[geom_id]
        geom_meshid = physics.model.geom_dataid[geom_id]

        collision_object.pose.position.x = geom_pos[0]
        collision_object.pose.position.y = geom_pos[1]
        collision_object.pose.position.z = geom_pos[2]
        collision_object.pose.orientation.w = geom_xquat[0]
        collision_object.pose.orientation.x = geom_xquat[1]
        collision_object.pose.orientation.y = geom_xquat[2]
        collision_object.pose.orientation.z = geom_xquat[3]

        if geom_type == mjtGeom.mjGEOM_MESH:
            mesh = Mesh()
            # TODO: implement me
            # mesh_name = mj_id2name(physics.model.ptr, mju_str2Type('mesh'), geom_meshid)
            # mesh_name = mesh_name.split("/")[1]  # skip the model prefix, e.g. val/my_mesh
            # collision_object.type = Marker.MESH_RESOURCE
            # collision_object.mesh_resource = f"package://dm_envs/meshes/{mesh_name}.stl"
            #
            # collision_object.scale.x = 1
            # collision_object.scale.y = 1
            # collision_object.scale.z = 1

            mesh_pose = Pose()
            mesh_pose.position.x = body_pos[0]
            mesh_pose.position.y = body_pos[1]
            mesh_pose.position.z = body_pos[2]
            mesh_pose.orientation.w = body_xquat[0]
            mesh_pose.orientation.x = body_xquat[1]
            mesh_pose.orientation.y = body_xquat[2]
            mesh_pose.orientation.z = body_xquat[3]

            collision_object.meshes.append(mesh)
            collision_object.mesh_poses.append(mesh_pose)
        else:
            primitive = SolidPrimitive()
            primitive_pose = Pose()
            if geom_type == mjtGeom.mjGEOM_BOX:
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = (geom_size * 2).tolist()
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                primitive.type = SolidPrimitive.CYLINDER
                primitive.dimensions = [geom_size[0] * 2, geom_size[0] * 2, geom_size[1] * 2]
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                primitive.type = SolidPrimitive.CYLINDER
                primitive.dimensions = [geom_size[0] * 2, geom_size[0] * 2, geom_size[1] * 2]
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                primitive.type = SolidPrimitive.SPHERE
                primitive.dimensions = [geom_size[0] * 2, geom_size[0] * 2, geom_size[0] * 2]
            else:
                rospy.loginfo_once(f"Unsupported geom type {geom_type}")
                continue

            collision_object.primitives.append(primitive)
            collision_object.primitive_poses.append(primitive_pose)

        msg.world.collision_objects.append(collision_object)

    return msg


class MoveitPlanningScenePublisher:

    def __init__(self, scene_topic: str = '/hdt_michigan/planning_scene'):
        self.pub = rospy.Publisher(scene_topic, PlanningScene, queue_size=10)

    def update(self, physics):
        msg = make_planning_scene_msg(physics)
        self.pub.publish(msg)


class VoxelgridBuild(composer.Entity):
    def _build(self, res: float):
        self._model = mjcf.element.RootElement(model='vgb_sphere')
        self._geom = self._model.worldbody.add('geom', name='geom', type='sphere', size=[res])

    @property
    def mjcf_model(self):
        return self._model


class StaticEnvEntity(composer.Entity):
    def _build(self, path: str):
        print(f"Loading {path}")
        self._model = mjcf.from_path(path)

    @property
    def mjcf_model(self):
        return self._model


class ValEntity(composer.Entity):
    def _build(self):
        self._model = mjcf.from_path('val_husky_no_gripper_collisions.xml')

    @property
    def mjcf_model(self):
        return self._model

    @property
    def joints(self):
        return self.mjcf_model.find_all('joint')

    @property
    def joint_names(self):
        return [j.name for j in self.joints]


class ValRopeManipulation(BaseRopeManipulation):

    def __init__(self, params: Dict):
        super().__init__(params)

        self.tfw = TF2Wrapper()
        self.psp = MoveitPlanningScenePublisher()

        # other entities
        self._val = ValEntity()
        self._static_env = StaticEnvEntity(params.get('static_env_filename', 'empty.xml'))
        self.vgb = VoxelgridBuild(res=0.01)

        val_site = self._arena.attach(self._val)
        val_site.pos = [0, 0, 0.0]
        static_env_site = self._arena.attach(self._static_env)
        static_env_site.pos = [1.22, -0.14, 0.1]
        static_env_site.quat = quaternion_from_euler(0, 0, -1.5707)
        self._arena.add_free_entity(self.vgb)

        self._arena.mjcf_model.equality.add('distance', name='left_grasp', geom1='val/left_tool_geom', geom2='rope/rG0',
                                            distance=0, active='false', solref="0.02 2")
        self._arena.mjcf_model.equality.add('distance', name='right_grasp', geom1='val/right_tool_geom',
                                            geom2=f'rope/rG{self.rope.length - 1}', distance=0, active='false',
                                            solref="0.02 2")

        self._actuators = self._arena.mjcf_model.find_all('actuator')

        self._task_observables.update({
            'left_gripper':    observable.MujocoFeature('site_xpos', 'val/left_tool'),
            'right_gripper':   observable.MujocoFeature('site_xpos', 'val/right_tool'),
            'joint_positions': observable.MJCFFeature('qpos', self.actuated_joints),
        })

        for obs_ in self._task_observables.values():
            obs_.enabled = True

    @property
    def joints(self):
        return self._val.joints

    @property
    def actuated_joints(self):
        return [a.joint for a in self._actuators]

    @property
    def joint_names(self):
        return [f'val/{n}' for n in self._val.joint_names]

    @property
    def actuated_joint_names(self):
        return [f'val/{a.joint.name}' for a in self._actuators]

    def initialize_episode(self, physics, random_state):
        with physics.reset_context():
            # this will overwrite the pose set when val is 'attach'ed to the arena
            self.tfw.send_transform(parent='world', child='robot_root', is_static=True,
                                    translation=[0, 0, 0.15], quaternion=[0, 0, 0, 1])
            self._val.set_pose(physics,
                               position=[0, 0, 0.15],
                               quaternion=quaternion_from_euler(0, 0, 0))
            self.rope.set_pose(physics,
                               position=[0.5, -self.rope.length_m / 2, 0.6],
                               quaternion=quaternion_from_euler(0, 0, 1.5707))
            for i in range(self.rope.length - 1):
                physics.named.data.qpos[f'rope/rJ1_{i + 1}'] = 0

    def current_action_vec(self, physics):
        left_tool_pos = physics.named.data.xpos['val/left_tool']
        right_tool_pos = physics.named.data.xpos['val/right_tool']
        right_tool_quat = physics.named.data.xquat['val/right_tool']
        left_tool_quat = physics.named.data.xquat['val/left_tool']
        return np.concatenate((left_tool_pos, left_tool_quat, right_tool_pos, right_tool_quat))

    def solve_ik(self, physics, target_pos, target_quat, site_name):
        # store the initial qpos to restore later
        initial_qpos = physics.bind(self.actuated_joints).qpos.copy()
        result = inverse_kinematics.qpos_from_site_pose(
            physics=physics,
            site_name=site_name,
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=self.actuated_joint_names,
            rot_weight=2,  # more rotation weight than the default
            # max_steps=10000,
            inplace=True,
        )
        qdes = physics.named.data.qpos[self.actuated_joint_names]
        # reset the arm joints to their original positions, because the above functions actually modify physics state
        physics.bind(self.actuated_joints).qpos = initial_qpos
        return result.success, qdes

    def release_rope(self, physics):
        physics.model.eq_active[:] = np.zeros(1)

    def grasp_rope(self, physics):
        physics.model.eq_active[:] = np.ones(1)

    def follow_trajectory(self, env, trajectory: JointTrajectory):
        # TODO: make this generic so it could be used by Andrea in pybullet
        traj_goal = make_follow_joint_trajectory_goal(trajectory)
        initial_joint_positions = env._observation_updater.get_observation()['joint_positions'].flatten()

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
            actual_joint_positions = self.get_joint_positions(env, trajectory_joint_names)

            actual_point = JointTrajectoryPoint(positions=actual_joint_positions, time_from_start=dt)
            while trajectory_point_idx < len(interpolated_points) - 1 and \
                    is_waypoint_reached(actual_point, interpolated_points[trajectory_point_idx], tolerance):
                trajectory_point_idx += 1

            desired_point = interpolated_points[trajectory_point_idx]

            if trajectory_point_idx >= len(interpolated_points) - 1 and \
                    is_waypoint_reached(actual_point, desired_point, goal_tolerance):
                return True

            action_vec = self.action_vec_from_positions_and_names(desired_point.positions, trajectory_joint_names,
                                                                  initial_joint_positions)
            env.step(action_vec)

            error = waypoint_error(actual_point, desired_point)
            # print(trajectory_point_idx, len(trajectory.points), f"{error} {desired_point.time_from_start.to_sec()} {dt.to_sec()}")
            if desired_point.time_from_start.to_sec() > 0 and dt > desired_point.time_from_start * 2.0:
                if trajectory_point_idx == len(interpolated_points) - 1:
                    print(f"timeout. expected t={desired_point.time_from_start.to_sec()} but t={dt.to_sec()}." \
                          + f" error to waypoint is {error}, goal tolerance is {goal_tolerance}")
                else:
                    print(f"timeout. expected t={desired_point.time_from_start.to_sec()} but t={dt.to_sec()}." \
                          + f" error to waypoint is {error}, tolerance is {tolerance}")
                return True

    def get_joint_positions(self, env, joint_names):
        obs = env._observation_updater.get_observation()
        actual_joint_positions = []
        for n in joint_names:
            # NOTE: this doesn't work for joints that aren't actuated? (leftgripper)
            i = self.actuated_joint_names.index(f'val/{n}')
            actual_joint_positions.append(obs['joint_positions'][0, i])
        return actual_joint_positions

    def action_vec_from_positions_and_names(self, positions, trajectory_joint_names, initial_joint_positions):
        action_vec = initial_joint_positions.copy()
        for i, a in enumerate(self._actuators):
            if a.joint.name in trajectory_joint_names:
                j = trajectory_joint_names.index(a.joint.name)
                action_vec[i] = positions[j]
        return action_vec

    def before_step(self, physics, action, random_state):
        self.psp.update(physics)
        super().before_step(physics, action, random_state)
        physics.set_control(action)


@ros_init.with_ros("val_rope_task")
def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=250)
    task = ValRopeManipulation({
        'max_step_size':       0.001,
        'static_env_filename': 'car1.xml',
    })
    env = composer.Environment(task, random_state=0, time_limit=9999)

    # from dm_control import viewer
    # viewer.launch(env)

    env.reset()

    from arm_robots.hdt_michigan import Val
    val = Val()
    val.set_execute(False)
    start_state = RobotState()
    start_state.joint_state.name = val.get_joint_names(group_name='both_arms')

    start_state.joint_state.position = task.get_joint_positions(env, start_state.joint_state.name)
    plan = val.plan_to_joint_config(group_name='both_arms',
                                    joint_config=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    start_state=start_state)
    task.follow_trajectory(env, plan.planning_result.plan.joint_trajectory)

    start_state.joint_state.position = task.get_joint_positions(env, start_state.joint_state.name)
    pose = Pose()
    pose.position.x = 0.8
    pose.position.y = -0.2
    pose.position.z = 0.4
    q = quaternion_from_euler(0, 0, 0)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    plan = val.plan_to_pose(group_name='right_side',
                            ee_link_name='right_tool',
                            target_pose=pose,
                            start_state=start_state)
    task.follow_trajectory(env, plan.planning_result.plan.joint_trajectory)


if __name__ == "__main__":
    main()
