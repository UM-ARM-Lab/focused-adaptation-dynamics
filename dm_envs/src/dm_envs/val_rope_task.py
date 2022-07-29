from typing import Dict

import mujoco
import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.mjcf import Physics
from dm_control.utils import inverse_kinematics
from transformations import quaternion_from_euler

import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from dm_envs.mujoco_services import my_step
from dm_envs.mujoco_visualizer import MujocoVisualizer
from dm_envs.base_rope_task import BaseRopeManipulation
from link_bot_pycommon.grid_utils_np import idx_to_point_3d_from_origin_point, vox_to_voxelgrid_stamped


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

        # other entities
        self._val = ValEntity()
        self._static_env = StaticEnvEntity(params.get('static_env_filename', 'empty.xml'))
        self.vgb = VoxelgridBuild(res=0.01)

        val_site = self._arena.attach(self._val)
        val_site.pos = [0, 0, 0.15]
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
            'left_gripper':       observable.MujocoFeature('site_xpos', 'val/left_tool'),
            'right_gripper':      observable.MujocoFeature('site_xpos', 'val/right_tool'),
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
            self._val.set_pose(physics,
                               position=[0, 0, 0.15],
                               quaternion=quaternion_from_euler(0, 0, 0))
            self.rope.set_pose(physics,
                               position=[0.5, -self.rope.length_m / 2, 0.5],
                               quaternion=quaternion_from_euler(0, 0, 1.5707))
            for i in range(self.rope.length - 1):
                physics.named.data.qpos[f'rope/rJ1_{i + 1}'] = 0

    def solve_ik(self, target_pos, target_quat, site_name):
        # store the initial qpos to restore later
        initial_qpos = env.physics.bind(task.actuated_joints).qpos.copy()
        result = inverse_kinematics.qpos_from_site_pose(
            physics=env.physics,
            site_name=site_name,
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=task.actuated_joint_names,
            rot_weight=2,  # more rotation weight than the default
            # max_steps=10000,
            inplace=True,
        )
        qdes = env.physics.named.data.qpos[task.actuated_joint_names]
        # reset the arm joints to their original positions, because the above functions actually modify physics state
        env.physics.bind(task.actuated_joints).qpos = initial_qpos
        return result.success, qdes

    def release_rope(self, physics):
        physics.model.eq_active[:] = np.zeros(1)

    def grasp_rope(self, physics):
        physics.model.eq_active[:] = np.ones(1)

    def before_step(self, physics: Physics, action, random_state):
        super().before_step(physics, action, random_state)

        from tqdm import tqdm
        from time import perf_counter
        from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

        pub = get_connected_publisher('/occupancy', VoxelgridStamped, queue_size=1)

        geom_type = mujoco.mju_str2Type('geom')

        def in_collision(xyz):
            self.vgb.set_pose(physics, position=xyz)
            mujoco.mj_step1(physics.model.ptr, physics.data.ptr)
            for i, c in enumerate(physics.data.contact):
                geom1_name = mujoco.mj_id2name(physics.model.ptr, geom_type, c.geom1)
                geom2_name = mujoco.mj_id2name(physics.model.ptr, geom_type, c.geom2)
                if c.dist < 0 and (geom1_name == 'vgb_sphere/geom' or geom2_name == 'vgb_sphere/geom'):
                    # print(f"Contact at {xyz} between {geom1_name} and {geom2_name}, {c.dist=:.4f} {c.exclude=}")
                    return True
            return False

        t0 = perf_counter()
        res = 0.02
        rows = 60
        columns = 50
        channels = 55
        vg = np.zeros([rows, columns, channels], dtype=np.float32)
        origin_point = np.array([0, -0.5, 0])

        for row, col, channel in tqdm(np.ndindex(rows, columns, channels), total=rows * columns * channels):
            xyz = idx_to_point_3d_from_origin_point(row, col, channel, res, origin_point)
            if in_collision(xyz):
                vg[row, col, channel] = 1
        print(perf_counter() - t0)

        pub.publish(vox_to_voxelgrid_stamped(vg, scale=res, frame='world'))

    def get_reward(self, physics):
        return 0


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, linewidth=250)
    task = ValRopeManipulation({
        'seconds_per_substep': 0.001,
        'static_env_filename': 'car1.xml',
    })
    env = composer.Environment(task, random_state=0)
    viz = MujocoVisualizer()

    # from dm_control import viewer
    # viewer.launch(env)

    rospy.init_node("val_rope_task")

    env.reset()

    # # move to grasp
    # _, qdes = task.solve_ik(target_pos=[0, task.rope.length_m / 2, 0.05],
    #                         target_quat=quaternion_from_euler(0, -np.pi, 0),
    #                         site_name='val/left_tool')
    # my_step(viz, env, qdes, 20)

    # grasp!
    task.grasp_rope(env.physics)
    my_step(viz, env, [0] * 20, 20)

    # # lift up
    # _, qdes = task.solve_ik(target_pos=[0, 0, 0.5],
    #                         target_quat=quaternion_from_euler(0, -np.pi - 0.4, 0),
    #                         site_name='val/left_tool')
    # my_step(viz, env, [0] * 20, 20)

    # release
    task.release_rope(env.physics)
    my_step(viz, env, [0] * 20, 20)
