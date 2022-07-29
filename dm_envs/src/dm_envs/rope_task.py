from typing import Dict

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation.observable import MujocoFeature
from dm_control.mjcf import Physics

import rospy
from dm_envs.base_rope_task import BaseRopeManipulation
from dm_envs.mujoco_services import my_step
from dm_envs.mujoco_visualizer import MujocoVisualizer

seed = 0


class GripperEntity(composer.Entity):
    def _build(self, name=None, rgba=(0, 1, 1, 1), mass=0.01):
        self._model = mjcf.RootElement(name)
        body = self._model.worldbody.add('body', name='body')
        body.add('geom', name='geom', size=[0.01, 0.01, 0.01], mass=mass, rgba=rgba, type='box', contype=2,
                 conaffinity=2, group=1)
        body.add('joint', axis=[1, 0, 0], type='slide', name='x', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[0, 1, 0], type='slide', name='y', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[0, 0, 1], type='slide', name='z', pos=[0, 0, 0], limited=False)

    @property
    def mjcf_model(self):
        return self._model


class RopeManipulation(BaseRopeManipulation):

    def __init__(self, params: Dict):
        super().__init__(params)

        self.left_gripper = GripperEntity(name='left_gripper', rgba=(0, 1, 1, 1), mass=0.01)
        self.right_gripper = GripperEntity(name='right_gripper', rgba=(0.5, 0, 0.5, 1), mass=0.1)

        left_gripper_site = self._arena.attach(self.left_gripper)
        self.left_gripper_initial_pos = np.array([-self.rope.half_capsule_length, 0, 0])
        left_gripper_site.pos = self.left_gripper_initial_pos
        right_gripper_site = self._arena.attach(self.right_gripper)
        self.right_gripper_initial_pose = [1 - self.rope.half_capsule_length, 0, 0]
        right_gripper_site.pos = self.right_gripper_initial_pose
        self.initial_action = np.concatenate((self.left_gripper_initial_pos, self.right_gripper_initial_pose))

        # constraint
        self._arena.mjcf_model.equality.add('connect', anchor=[0, 0, 0], body1='left_gripper/body', body2='rope/rB0',
                                            solref='0.02 1')
        self._arena.mjcf_model.equality.add('connect', anchor=[0, 0, 0], body1='right_gripper/body',
                                            body2=f'rope/rB{self.rope.length - 1}', solref='0.02 1')

        # actuators
        def _make_actuator(joint, name):
            joint.damping = 10
            self._arena.mjcf_model.actuator.add('position',
                                                name=name,
                                                joint=joint,
                                                forcelimited=True,
                                                forcerange=[-100, 100],
                                                ctrllimited=False,
                                                kp=10)

        _make_actuator(self.left_gripper.mjcf_model.find_all('joint')[0], 'left_gripper_x')
        _make_actuator(self.left_gripper.mjcf_model.find_all('joint')[1], 'left_gripper_y')
        _make_actuator(self.left_gripper.mjcf_model.find_all('joint')[2], 'left_gripper_z')
        _make_actuator(self.right_gripper.mjcf_model.find_all('joint')[0], 'right_gripper_x')
        _make_actuator(self.right_gripper.mjcf_model.find_all('joint')[1], 'right_gripper_y')
        _make_actuator(self.right_gripper.mjcf_model.find_all('joint')[2], 'right_gripper_z')
        self._actuators = self._arena.mjcf_model.find_all('actuator')

        self._task_observables.update({
            'left_gripper':  MujocoFeature('geom_xpos', 'left_gripper/geom'),
            'right_gripper': MujocoFeature('geom_xpos', 'right_gripper/geom'),
        })

        for obs_ in self._task_observables.values():
            obs_.enabled = True

    def initialize_episode(self, physics, random_state):
        x = 0
        y = 0
        z = 0.01
        with physics.reset_context():
            self.rope.set_pose(physics, position=(x + self.rope.half_capsule_length, y, z))
            self.left_gripper.set_pose(physics, position=(x, y, z))
            self.right_gripper.set_pose(physics, position=(x + self.rope.length_m, y, z))
            for i in range(10):
                physics.named.data.qpos[f'rope/rJ1_{i + 1}'] = 0

    def before_step(self, physics: Physics, action, random_state):
        relative_action = action - self.initial_action
        physics.set_control(relative_action)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, linewidth=250)
    task = RopeManipulation({})
    env = composer.Environment(task)
    viz = MujocoVisualizer()

    rospy.init_node("rope_task")

    # my_step(viz, env, np.array([0] * 6), 20)

    my_step(viz, env, np.array([0, 0, 0.5, 0.5, 0, 0.5]), 20)

    my_step(viz, env, np.array([0, 0.25, 0.5, 0.5, -0.25, 0.5]), 100)
