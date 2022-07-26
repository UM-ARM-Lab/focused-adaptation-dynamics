from dm_control import composer, viewer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from dm_control.utils import inverse_kinematics

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
        self._thickness = thickness
        self._spacing = length_m / length
        self.half_capsule_length = length_m / (length * 2)
        self._model = mjcf.RootElement('rope')
        self._model.compiler.angle = 'radian'
        body = self._model.worldbody.add('body', name='rB0')
        self._composite = body.add('composite', prefix="r", type='rope', count=[length, 1, 1], spacing=self._spacing)
        self._composite.add('joint', kind='main', damping=1e-2, stiffness=stiffness)
        self._composite.geom.set_attributes(type='capsule', size=[self._thickness, self.half_capsule_length],
                                            rgba=rgba, mass=0.005, contype=1, conaffinity=1, priority=1,
                                            friction=[0.1, 5e-3, 1e-4])

    @property
    def mjcf_model(self):
        return self._model


class RopeManipulation(composer.Task):
    NUM_SUBSTEPS = 10  # The number of physics substeps per control timestep.

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

        # other entities
        self._val = ValEntity()
        self._rope = RopeEntity(length=rope_length)

        self._arena.add_free_entity(self._rope)
        # self._arena.add_free_entity(self._val)
        val_site = self._arena.attach(self._val) # if you want val to be fixed to the world
        val_site.pos = [-1, 0, 0.15]

        # constraint
        # self._arena.mjcf_model.equality.add('connect', body1='val/right_tool', body2='rope/rB0', anchor=[0, 0, 0])
        # self._arena.mjcf_model.equality.add('connect', body1='val/right_tool',
        #                                     body2=f'rope/rB{self._rope.length - 1}',
        #                                     anchor=[0, 0, 0])
        self._actuators = self._arena.mjcf_model.find_all('actuator')

        self._task_observables = {
            'rope_pos':   observable.MujocoFeature('geom_xpos', [f'rope/rG{i}' for i in range(rope_length)]),
            'left_tool':  observable.MujocoFeature('site_xpos', 'val/left_tool'),
            'right_tool': observable.MujocoFeature('site_xpos', 'val/right_tool'),
            # 'joint_positions': observable.MujocoFeature('qpos', 'val'),
        }
        for a in self._actuators:
            self._task_observables[a.joint.name] = observable.MujocoFeature('qpos', f'val/{a.joint.name}')

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

    def initialize_episode_mjcf(self, random_state):
        pass

    def initialize_episode(self, physics, random_state):
        with physics.reset_context():
            self._val.set_pose(physics, position=[-1, 0, 0.15])
            for i in range(self._rope.length - 1):
                physics.named.data.qpos[f'rope/rJ1_{i + 1}'] = 0

    def before_step(self, physics, action, random_state):
        physics.set_control(action)

    def get_reward(self, physics):
        return 0


if __name__ == "__main__":
    task = RopeManipulation()
    seed = None
    env = composer.Environment(task, random_state=seed)
    obs = env.reset()

    i = 0

    # solve IK
    initial_qpos = env.physics.bind(task.actuated_joints).qpos.copy()

    result = inverse_kinematics.qpos_from_site_pose(
        physics=env.physics,
        site_name=f'val/left_tool',
        target_pos=[0, 0, 0.5],
        target_quat=[0, 0, 0, 1],
        joint_names=task.actuated_joint_names,
        # rot_weight=2,  # more rotation weight than the default
        inplace=True,
    )

    qdes = env.physics.named.data.qpos[task.actuated_joint_names]

    # reset the arm joints to their original positions, because the above functions actually modify physics state
    env.physics.bind(task.actuated_joints).qpos = initial_qpos

    print('ik success?', result.success)


    def random_policy(time_step):
        # return qdes
        return [0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    viewer.launch(env, policy=random_policy)
    # steps_per_second = int(1 / task.control_timestep)
    # action = [0.8, 0, 1, 0, 0, 1]
    #
    # from time import perf_counter
    #
    # t0 = perf_counter()
    # sim_seconds = 10
    # for i in range(steps_per_second * sim_seconds):
    #     time_step = env.step(action)
    # real_seconds = perf_counter() - t0
    # print(sim_seconds / real_seconds)
