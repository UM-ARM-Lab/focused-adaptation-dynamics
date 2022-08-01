from typing import Dict

from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors

from dm_envs.mujoco_visualizer import MujocoVisualizer


class BaseRopeManipulation(composer.Task):
    NUM_SUBSTEPS = 20  # The number of physics substeps per control timestep.

    def __init__(self, params: Dict):
        self.use_viz = params.get("use_viz", True)

        rope_length = params.get('rope_length', 25)
        seconds_per_substep = params.get('max_step_size', 0.01)
        # root entity
        self._arena = floors.Floor()
        self._arena.mjcf_model.worldbody.add('camera', name="mycamera", mode='fixed', pos=[0.5, -2, 2],
                                             euler=[1, -.1, 0])

        self._viz = MujocoVisualizer()

        # simulation setting
        self._arena.mjcf_model.compiler.inertiafromgeom = True
        self._arena.mjcf_model.default.joint.damping = 0
        self._arena.mjcf_model.default.joint.stiffness = 0
        self._arena.mjcf_model.default.material.reflectance = 0
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
            'rope': observable.MujocoFeature('geom_xpos', [f'rope/rG{i}' for i in range(rope_length)]),
        }

        for obs_ in self._task_observables.values():
            obs_.enabled = True

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def get_reward(self, physics):
        return 0

    def viz(self, physics):
        self._viz.viz(physics)

    def before_step(self, physics, action, random_state):
        if self.use_viz:
            self.viz(physics)


class RopeEntity(composer.Entity):
    def _build(self, length=25, length_m=1, rgba=(0.2, 0.8, 0.2, 1), thickness=0.01, stiffness=0.001, mass=0.2):
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
        self._composite.add('joint', kind='twist', damping=1e-3, stiffness=1e-4)
        self._composite.geom.set_attributes(type='capsule', size=[self.thickness, self.half_capsule_length],
                                            rgba=rgba, mass=mass / length, friction=[0.1, 5e-3, 1e-4])

    @property
    def mjcf_model(self):
        return self._model
