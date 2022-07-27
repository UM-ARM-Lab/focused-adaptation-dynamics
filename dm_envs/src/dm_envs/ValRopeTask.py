import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control import viewer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control.utils import inverse_kinematics
from transformations import quaternion_from_euler

seed = 0


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


class RopeManipulation(composer.Task):
    NUM_SUBSTEPS = 100  # The number of physics substeps per control timestep.

    def __init__(self, rope_length=25, seconds_per_substep=0.001):
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
        self._arena.mjcf_model.size.nconmax = 10000
        self._arena.mjcf_model.size.njmax = 10000

        # other entities
        self._val = ValEntity()
        self.rope = RopeEntity(length=rope_length)

        self._arena.add_free_entity(self.rope)
        # self._arena.add_free_entity(self._val)
        val_site = self._arena.attach(self._val)  # if you want val to be fixed to the world
        val_site.pos = [0, 0, 0.15]

        self._arena.mjcf_model.equality.add('distance', name='left_grasp', geom1='val/left_tool_geom', geom2='rope/rG0',
                                            distance=0, active='false', solref="0.02 2")
        self._arena.mjcf_model.equality.add('distance', name='right_grasp', geom1='val/right_tool_geom',
                                            geom2=f'rope/rG{self.rope.length - 1}', distance=0, active='false',
                                            solref="0.02 2")

        self._actuators = self._arena.mjcf_model.find_all('actuator')

        self._task_observables = {
            'rope_pos':        observable.MujocoFeature('geom_xpos', [f'rope/rG{i}' for i in range(rope_length)]),
            'left_tool':       observable.MujocoFeature('site_xpos', 'val/left_tool'),
            'right_tool':      observable.MujocoFeature('site_xpos', 'val/right_tool'),
            'joint_positions': observable.MJCFFeature('qpos', self.actuated_joints),
        }

        for obs_ in self._task_observables.values():
            obs_.enabled = True

        self.control_timestep = self.NUM_SUBSTEPS * self.physics_timestep

    @property
    def root_entity(self):
        return self._arena

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

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode(self, physics, random_state):
        with physics.reset_context():
            # this will overrite the pose set when val is 'attach'ed to the arena
            self._val.set_pose(physics,
                               position=[-0.8, 0, 0.15],
                               quaternion=quaternion_from_euler(0, 0, 0))
            for i in range(self.rope.length - 1):
                physics.named.data.qpos[f'rope/rJ1_{i + 1}'] = 0

    def before_step(self, physics, action, random_state):
        physics.set_control(action)

    def get_reward(self, physics):
        return 0

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


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, linewidth=250)
    task = RopeManipulation()
    env = composer.Environment(task)


    def step(action, n_steps=1):
        for i in range(n_steps):
            yield action


    def controller_gen():
        while True:
            # move to grasp
            _, qdes = task.solve_ik(target_pos=[0, 0, 0.05],
                                    target_quat=quaternion_from_euler(0, -np.pi, 0),
                                    site_name='val/left_tool')
            yield from step(qdes, 20)

            # grasp!
            task.grasp_rope(env.physics)

            yield from step(qdes, 20)

            # lift up
            _, qdes = task.solve_ik(target_pos=[0, 0, 0.5],
                                    target_quat=quaternion_from_euler(0, -np.pi - 0.4, 0),
                                    site_name='val/left_tool')

            yield from step([0] * 20, 20)

            task.release_rope(env.physics)

            yield from step([0] * 20, 20)


    c = controller_gen()

    viewer.launch(env, policy=lambda _: next(c))
