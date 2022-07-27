from dm_control import composer

from dm_envs.mujoco_visualizer import MujocoVisualizer
from link_bot_pycommon.base_services import BaseServices


class MujocoServices(BaseServices):

    def __init__(self):
        super().__init__()

    def setup_env(self, *args, **kwargs):
        pass


def my_step(viz: MujocoVisualizer, env: composer.Environment, action, n_steps=1):
    for i in range(n_steps):
        viz.viz(env.physics)
        time_step = env.step(action)
    return time_step.observation
