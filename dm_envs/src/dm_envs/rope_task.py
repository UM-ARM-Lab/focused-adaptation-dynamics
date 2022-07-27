from typing import Dict

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors

import rospy
from dm_envs.mujoco_services import my_step
from dm_envs.mujoco_visualizer import MujocoVisualizer

seed = 0


class RopeEntity(composer.Entity):
    def _build(self, length=25, length_m=1, rgba=(0.2, 0.8, 0.2, 1), thickness=0.01, stiffness=0.01):
        self.length = length
        self.length_m = length_m
        self.thickness = thickness
        self._spacing = length_m / length
        self.half_capsule_length = length_m / (length * 2)
        self._model = mjcf.RootElement('rope')
        self._model.compiler.angle = 'radian'
        body = self._model.worldbody.add('body', name='rB0')
        self._composite = body.add('composite', prefix="r", type='rope', count=[length, 1, 1], spacing=self._spacing)
        self._composite.add('joint', kind='main', damping=1e-2, stiffness=stiffness)
        self._composite.geom.set_attributes(type='capsule', size=[self.thickness, self.half_capsule_length],
                                            rgba=rgba, mass=0.005, contype=1, conaffinity=1, priority=1,
                                            friction=[0.1, 5e-3, 1e-4])

    @property
    def mjcf_model(self):
        return self._model


class GripperEntity(composer.Entity):
    def _build(self, name=None, rgba=(0, 1, 1, 1), mass=0.01):
        self._model = mjcf.RootElement(name)
        body = self._model.worldbody.add('body', name='dummy')
        body.add('geom', size=[0.01, 0.01, 0.01], mass=mass, rgba=rgba, type='box', contype=2, conaffinity=2, group=1)
        body.add('joint', axis=[1, 0, 0], type='slide', name='x', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[0, 1, 0], type='slide', name='y', pos=[0, 0, 0], limited=False)
        body.add('joint', axis=[0, 0, 1], type='slide', name='z', pos=[0, 0, 0], limited=False)

    @property
    def mjcf_model(self):
        return self._model


class BaseRopeManipulation(composer.Task):
    NUM_SUBSTEPS = 100  # The number of physics substeps per control timestep.

    def __init__(self, params: Dict):
        rope_length = params.get('rope_length', 25)
        seconds_per_substep = params.get('seconds_per_substep', 0.01)
        # root entity
        self._arena = floors.Floor()

        # simulation setting
        self._arena.mjcf_model.compiler.inertiafromgeom = True
        self._arena.mjcf_model.default.joint.damping = 0
        self._arena.mjcf_model.default.joint.stiffness = 0
        self._arena.mjcf_model.default.geom.contype = 3
        self._arena.mjcf_model.default.geom.conaffinity = 3
        self._arena.mjcf_model.default.geom.friction = [1, 0.1, 0.1]
        self._arena.mjcf_model.option.gravity = [0, 0, -9.81]
        self._arena.mjcf_model.option.integrator = 'Euler'
        self._arena.mjcf_model.option.timestep = seconds_per_substep
        self._arena.mjcf_model.size.nconmax = 1_000
        self.control_timestep = self.NUM_SUBSTEPS * self.physics_timestep

        # other entities
        self.rope = RopeEntity(length=rope_length)

        self._arena.add_free_entity(self.rope)

        self._task_observables = {
            'rope_pos': observable.MujocoFeature('geom_xpos', [f'rope/rG{i}' for i in range(rope_length)]),
        }

        for obs_ in self._task_observables.values():
            obs_.enabled = True

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def before_step(self, physics, action, random_state):
        physics.set_control(action)

    def get_reward(self, physics):
        return 0


class RopeManipulation(BaseRopeManipulation):

    def __init__(self, params: Dict):
        super().__init__(params)

        self._gripper1 = GripperEntity(name='gripper1', rgba=(0, 1, 1, 1), mass=0.01)
        self._gripper2 = GripperEntity(name='gripper2', rgba=(0.5, 0, 0.5, 1), mass=0.1)

        gripper1_site = self._arena.attach(self._gripper1)
        gripper1_site.pos = [-self.rope.half_capsule_length, 0, 0]
        gripper2_site = self._arena.attach(self._gripper2)
        gripper2_site.pos = [1 - self.rope.half_capsule_length, 0, 0]

        # constraint
        self._arena.mjcf_model.equality.add('connect', body1='gripper1/dummy', body2='rope/rB0', anchor=[0, 0, 0])
        self._arena.mjcf_model.equality.add('connect', body1='gripper2/dummy', body2=f'rope/rB{self.rope.length - 1}',
                                            anchor=[0, 0, 0])

        # actuators
        def _make_actuator(joint, name):
            joint.damping = 10
            self._arena.mjcf_model.actuator.add('position',
                                                name=name,
                                                joint=joint,
                                                forcelimited=True,
                                                forcerange=[-100, 100],
                                                ctrllimited=False,
                                                kp=50)

        _make_actuator(self._gripper1.mjcf_model.find_all('joint')[0], 'gripper1_x')
        _make_actuator(self._gripper1.mjcf_model.find_all('joint')[1], 'gripper1_y')
        _make_actuator(self._gripper1.mjcf_model.find_all('joint')[2], 'gripper1_z')
        _make_actuator(self._gripper2.mjcf_model.find_all('joint')[0], 'gripper2_x')
        _make_actuator(self._gripper2.mjcf_model.find_all('joint')[1], 'gripper2_y')
        _make_actuator(self._gripper2.mjcf_model.find_all('joint')[2], 'gripper2_z')
        self._actuators = self._arena.mjcf_model.find_all('actuator')

    def initialize_episode(self, physics, random_state):
        x = 0
        y = 0
        z = 0
        with physics.reset_context():
            self.rope.set_pose(physics, position=(x + self.rope.half_capsule_length, y, z))
            self._gripper1.set_pose(physics, position=(x, y, z))
            self._gripper2.set_pose(physics, position=(x + self.rope.length_m, y, z))
            for i in range(10):
                physics.named.data.qpos[f'rope/rJ1_{i + 1}'] = 0


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, linewidth=250)
    task = RopeManipulation({})
    env = composer.Environment(task)
    viz = MujocoVisualizer()

    rospy.init_node("rope_task")

    my_step(viz, env, [0] * 6, 200)

    my_step(viz, env, [0, 0, 0.3, -0.5, 0, 0.3], 200)
