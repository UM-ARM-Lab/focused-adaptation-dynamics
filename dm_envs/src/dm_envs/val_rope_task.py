from typing import Dict, List, Optional

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
from dm_envs.abstract_follow_trajectory import follow_trajectory
from dm_envs.base_rope_task import BaseRopeManipulation
from geometry_msgs.msg import Pose
from moveit_msgs.msg import PlanningScene, CollisionObject
from moveit_msgs.srv import GetPlanningScene
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory


def make_planning_scene_msg(physics, exclude, initial_msg):
    initial_msg.is_diff = False
    initial_msg.world.collision_objects = []

    initial_msg.robot_state.joint_state.header.stamp = rospy.Time.now()
    initial_msg.robot_state.joint_state.position = []
    initial_msg.robot_state.joint_state.velocity = []
    initial_msg.robot_state.joint_state.effort = []
    for n in initial_msg.robot_state.joint_state.name:
        qname = f'val/{n}'
        if qname in physics.named.data.qpos.axes[0].names:
            p = float(physics.named.data.qpos[qname])
            v = float(physics.named.data.qvel[qname])
            a = float(physics.named.data.qacc[qname])
        else:
            p = 0
            v = 0
            a = 0
        initial_msg.robot_state.joint_state.position.append(p)
        initial_msg.robot_state.joint_state.velocity.append(v)
        initial_msg.robot_state.joint_state.effort.append(a)

    for geom_id in range(physics.model.ngeom):
        geom_name = mj_id2name(physics.model.ptr, mju_str2Type('geom'), geom_id)

        geom_bodyid = physics.model.geom_bodyid[geom_id]
        body_name = mj_id2name(physics.model.ptr, mju_str2Type('body'), geom_bodyid)

        skip = False
        for exclude_i in exclude:
            if exclude_i in body_name or exclude_i in geom_name:
                skip = True

        if skip:
            continue

        collision_object = CollisionObject()
        collision_object.header.frame_id = 'world'  # must match robot root link
        collision_object.header.stamp = rospy.Time.now()
        collision_object.operation = CollisionObject.ADD
        collision_object.id = f'{body_name}-{geom_name}'

        geom_type = physics.model.geom_type[geom_id]
        geom_pos = physics.data.geom_xpos[geom_id]
        geom_xmat = physics.data.geom_xmat[geom_id]
        geom_quat = np.zeros(4)
        mju_mat2Quat(geom_quat, geom_xmat)
        geom_size = physics.model.geom_size[geom_id]

        collision_object.pose.orientation.w = 1

        if geom_type == mjtGeom.mjGEOM_MESH:
            # not implementing this yet since the only meshes are those on the robot itself
            continue
        else:
            primitive = SolidPrimitive()
            primitive_pose = Pose()
            primitive_pose.position.x = geom_pos[0]
            primitive_pose.position.y = geom_pos[1]
            primitive_pose.position.z = geom_pos[2]
            primitive_pose.orientation.w = geom_quat[0]
            primitive_pose.orientation.x = geom_quat[1]
            primitive_pose.orientation.y = geom_quat[2]
            primitive_pose.orientation.z = geom_quat[3]
            if geom_type == mjtGeom.mjGEOM_BOX:
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = (geom_size * 2).tolist()
            elif geom_type == mjtGeom.mjGEOM_CYLINDER:
                primitive.type = SolidPrimitive.CYLINDER
                primitive.dimensions = [geom_size[0] * 2, geom_size[0]]
            elif geom_type == mjtGeom.mjGEOM_CAPSULE:
                primitive.type = SolidPrimitive.CYLINDER
                primitive.dimensions = [geom_size[0] * 2, geom_size[0]]
            elif geom_type == mjtGeom.mjGEOM_SPHERE:
                primitive.type = SolidPrimitive.SPHERE
                primitive.dimensions = [geom_size[0] * 2]
            else:
                rospy.loginfo_once(f"Unsupported geom type {geom_type}")
                continue

            collision_object.primitives.append(primitive)
            collision_object.primitive_poses.append(primitive_pose)

        initial_msg.world.collision_objects.append(collision_object)

    return initial_msg


class MoveitPlanningScenePublisher:

    def __init__(self, scene_topic: str = '/hdt_michigan/planning_scene', exclude: List[str] = None):
        self.latest_msg = None
        self.exclude = exclude if exclude is not None else []
        self.exclude.extend(['val', 'rope', 'vgb_sphere'])
        self.pub = rospy.Publisher(scene_topic, PlanningScene, queue_size=10)
        self.get_srv = rospy.ServiceProxy("/hdt_michigan/get_planning_scene", GetPlanningScene)

        # we get the ACM, link padding, etc... from this, then we add on top the collision objects and robot state
        planning_scene_response = self.get_srv()
        self.initial_scene_from_move_group = planning_scene_response.scene

    def update(self, physics):
        self.latest_msg = make_planning_scene_msg(physics, self.exclude, self.initial_scene_from_move_group)
        self.pub.publish(self.latest_msg)

    def get_planning_scene_msg(self, physics):
        if self.latest_msg is None:
            self.latest_msg = make_planning_scene_msg(physics, self.exclude, self.initial_scene_from_move_group)
        return self.latest_msg


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
        self.val_init_pos = np.array([0, 0, 0.15])
        val_site.pos = self.val_init_pos
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

        self.js_pub = rospy.Publisher("/hdt_michigan/joint_states", JointState, queue_size=10)

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
            self._val.set_pose(physics, position=self.val_init_pos)
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
        def _get_joint_positions(joint_names: Optional[List[str]] = None) -> np.ndarray:
            return self.get_joint_positions(env, joint_names)

        def _command_and_simulate(desired_point, trajectory_joint_names, initial_joint_positions):
            action_vec = self.action_vec_from_positions_and_names(desired_point.positions, trajectory_joint_names,
                                                                  initial_joint_positions)
            env.step(action_vec)

        follow_trajectory(trajectory, _get_joint_positions, _command_and_simulate)
        # let things settle at the end
        for i in range(40):
            env.step(None)

    def get_joint_positions(self, env, joint_names=None):
        obs = env._observation_updater.get_observation()
        actual_joint_positions = []

        if joint_names is None:
            return obs['joint_positions'].flatten()
        else:
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
        self.js_pub.publish(self.get_planning_scene_msg(physics).robot_state.joint_state)
        self.tfw.send_transform(parent='world', child='robot_root', is_static=True,
                                translation=self.val_init_pos, quaternion=[0, 0, 0, 1])
        self.tfw.send_transform(parent='world', child='base_link', is_static=True,
                                translation=self.val_init_pos, quaternion=[0, 0, 0, 1])
        if action is not None:
            physics.set_control(action)

    def get_planning_scene_msg(self, physics):
        return self.psp.get_planning_scene_msg(physics)


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

    while True:
        start_scene = task.get_planning_scene_msg(env.physics)
        plan = val.plan_to_joint_config(group_name='both_arms',
                                        joint_config=[0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                                        start_state=start_scene.robot_state)
        task.follow_trajectory(env, plan.planning_result.plan.joint_trajectory)

        start_scene = task.get_planning_scene_msg(env.physics)

        start_scene = task.get_planning_scene_msg(env.physics)
        plan = val.plan_to_joint_config(group_name='both_arms',
                                        joint_config=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        start_state=start_scene.robot_state)
        task.follow_trajectory(env, plan.planning_result.plan.joint_trajectory)

        pose = Pose()
        pose.position.x = 0.8
        pose.position.y = -0.2
        pose.position.z = 0.4
        q = quaternion_from_euler(0, 0, 0)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        start_scene = task.get_planning_scene_msg(env.physics)
        plan = val.plan_to_pose(group_name='right_side',
                                ee_link_name='right_tool',
                                target_pose=pose,
                                start_state=start_scene.robot_state)
        task.follow_trajectory(env, plan.planning_result.plan.joint_trajectory)

        rospy.sleep(5)

        start_scene = task.get_planning_scene_msg(env.physics)
        val.store_current_tool_orientations([val.right_tool_name])
        plan = val.follow_jacobian_to_position_from_scene_and_state(start_scene,
                                                                    start_scene.robot_state.joint_state,
                                                                    'both_arms',
                                                                    [val.right_tool_name],
                                                                    [[[0.8, -0.2, 0.6]]],
                                                                    vel_scaling=1.0)
        task.follow_trajectory(env, plan.planning_result.plan.joint_trajectory)

        rospy.sleep(5)


if __name__ == "__main__":
    main()
