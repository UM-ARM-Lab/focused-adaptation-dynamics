from typing import Dict

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation.observable import MujocoFeature
from dm_control.mjcf import Physics
from transformations import euler_from_quaternion

import rospy
from dm_envs.base_rope_task import BaseRopeManipulation
from dm_envs.mujoco_services import my_step
from dm_envs.mujoco_visualizer import MujocoVisualizer

seed = 0


class GripperEntity(composer.Entity):
    def _build(self, name=None, rgba=(0, 1, 1, 1), mass=0.01):
        self._model = mjcf.RootElement(name)
        body = self._model.worldbody.add('body', name='body')
        body.add('geom', name='geom', size=[0.02, 0.02, 0.02], mass=mass, rgba=rgba, type='box')
        body.add('joint', axis=[1, 0, 0], type='slide', name='x', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[0, 1, 0], type='slide', name='y', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[0, 0, 1], type='slide', name='z', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[1, 0, 0], type='hinge', name='r', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[0, 1, 0], type='hinge', name='p', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[0, 0, 1], type='hinge', name='t', pos=[0, 0, 0], limited=False)

    @property
    def mjcf_model(self):
        return self._model


class RopeManipulation(BaseRopeManipulation):

    def __init__(self, params: Dict):
        super().__init__(params)
        self.post_reset_action = None

        self.left_gripper = GripperEntity(name='left_gripper', rgba=(0, 1, 1, 1))
        self.right_gripper = GripperEntity(name='right_gripper', rgba=(0.5, 0, 0.5, 1))

        # we have to set the gripper positions here so that the rope and grippers connect with no offset
        left_gripper_site = self._arena.attach(self.left_gripper)
        gripper_offset = 0.05
        self.left_gripper_initial_pos = np.array([-self.rope.half_capsule_length - gripper_offset, 0, 0])
        left_gripper_site.pos = self.left_gripper_initial_pos
        right_gripper_site = self._arena.attach(self.right_gripper)
        self.right_gripper_initial_pos = [self.rope.length_m - self.rope.half_capsule_length + gripper_offset, 0, 0]
        right_gripper_site.pos = self.right_gripper_initial_pos

        # constraint
        self._arena.mjcf_model.equality.add('weld', body1='left_gripper/body', body2='rope/rB0')
        self._arena.mjcf_model.equality.add('weld', body1='right_gripper/body', body2=f'rope/rB{self.rope.length - 1}')

        # actuators
        def _make_pos_actuator(joint, name):
            joint.damping = 1
            self._arena.mjcf_model.actuator.add('position',
                                                name=name,
                                                joint=joint,
                                                forcelimited=True,
                                                forcerange=[-1, 1],
                                                ctrllimited=False,
                                                kp=1)

        def _make_rot_actuator(joint, name):
            joint.damping = 1
            self._arena.mjcf_model.actuator.add('position',
                                                name=name,
                                                joint=joint,
                                                forcelimited=True,
                                                forcerange=[-1, 1],
                                                ctrllimited=False,
                                                kp=0)

        _make_pos_actuator(self.left_gripper.mjcf_model.find_all('joint')[0], 'left_gripper_x')
        _make_pos_actuator(self.left_gripper.mjcf_model.find_all('joint')[1], 'left_gripper_y')
        _make_pos_actuator(self.left_gripper.mjcf_model.find_all('joint')[2], 'left_gripper_z')
        _make_rot_actuator(self.left_gripper.mjcf_model.find_all('joint')[3], 'left_gripper_r')
        _make_rot_actuator(self.left_gripper.mjcf_model.find_all('joint')[4], 'left_gripper_p')
        _make_rot_actuator(self.left_gripper.mjcf_model.find_all('joint')[5], 'left_gripper_t')
        _make_pos_actuator(self.right_gripper.mjcf_model.find_all('joint')[0], 'right_gripper_x')
        _make_pos_actuator(self.right_gripper.mjcf_model.find_all('joint')[1], 'right_gripper_y')
        _make_pos_actuator(self.right_gripper.mjcf_model.find_all('joint')[2], 'right_gripper_z')
        _make_rot_actuator(self.right_gripper.mjcf_model.find_all('joint')[3], 'right_gripper_r')
        _make_rot_actuator(self.right_gripper.mjcf_model.find_all('joint')[4], 'right_gripper_p')
        _make_rot_actuator(self.right_gripper.mjcf_model.find_all('joint')[5], 'right_gripper_t')
        self._actuators = self._arena.mjcf_model.find_all('actuator')

        self._task_observables.update({
            'left_gripper':  MujocoFeature('xpos', 'left_gripper/body'),
            'right_gripper': MujocoFeature('xpos', 'right_gripper/body'),
        })

        for obs_ in self._task_observables.values():
            obs_.enabled = True

    def initialize_episode(self, physics, random_state):
        z = 0.1
        with physics.reset_context():
            self.left_gripper.set_pose(physics, position=self.left_gripper_initial_pos + np.array([0, 0, z]))
            self.right_gripper.set_pose(physics, position=self.right_gripper_initial_pos + np.array([0, 0, z]))
            self.rope.set_pose(physics, position=np.array([0, 0, z]))

        self.post_reset_action = self.current_action_vec(physics)

        self.viz(physics)

    def current_action_vec(self, physics):
        left_gripper_pos = physics.named.data.xpos['left_gripper/body']
        right_gripper_pos = physics.named.data.xpos['right_gripper/body']
        right_gripper_euler = euler_from_quaternion(physics.named.data.xquat['right_gripper/body'])
        left_gripper_euler = euler_from_quaternion(physics.named.data.xquat['left_gripper/body'])
        return np.concatenate((left_gripper_pos, left_gripper_euler, right_gripper_pos, right_gripper_euler))

    def before_step(self, physics: Physics, action, random_state):
        relative_action = action - self.post_reset_action
        physics.set_control(relative_action)
        self.viz(physics)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, linewidth=250)
    task = RopeManipulation({})
    env = composer.Environment(task)
    viz = MujocoVisualizer()

    rospy.init_node("rope_task")

    env.reset()

    def _a(lleft_gripper_pos, left_gripper_euler, right_gripper_pos, right_gripper_euler):
        return np.concatenate((lleft_gripper_pos, left_gripper_euler, right_gripper_pos, right_gripper_euler))

    obs = my_step(task, env, _a([0, 0, 0.3], [0, np.pi / 2, 0], [0.5, 0, 0.3], [0, -np.pi / 2, 0]), 20)
    print(obs['left_gripper'], np.linalg.norm(obs['left_gripper'] - np.array([0, 0, 0.3])))
    print(obs['right_gripper'], np.linalg.norm(obs['right_gripper'] - np.array([0.5, 0, 0.3])))

    # obs = my_step(viz, env, np.array([0, 0.25, 0.5, 0.5, -0.25, 0.5]), 20)
    # print(obs['left_gripper'] - np.array([0, 0.25, 0.5]),
    #       np.linalg.norm(obs['left_gripper'] - np.array([0, 0.25, 0.5])))
    #
    # obs = my_step(viz, env, np.array([0.25, 0.25, 0.5, 0.5, -0.25, 0.5]), 20)
    # print(obs['left_gripper'] - np.array([0.25, 0.25, 0.5]),
    #       np.linalg.norm(obs['left_gripper'] - np.array([0.25, 0.25, 0.5])))
    #
    # obs = my_step(viz, env, np.array([0.25, 0.25, 0.25, 0.5, -0.25, 0.5]), 20)
    # print(obs['left_gripper'] - np.array([0.25, 0.25, 0.25]),
    #       np.linalg.norm(obs['left_gripper'] - np.array([0.25, 0.25, 0.25])))
