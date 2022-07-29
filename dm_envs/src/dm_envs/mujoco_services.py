import numpy as np
from dm_control import composer

from dm_envs.mujoco_visualizer import MujocoVisualizer
from link_bot_pycommon.base_services import BaseServices


class MujocoServices(BaseServices):

    def __init__(self):
        super().__init__()

    def setup_env(self, *args, **kwargs):
        pass


def my_step(viz: MujocoVisualizer, env: composer.Environment, action, n_steps=1):
    obs = env._observation_updater.get_observation()
    initial = np.concatenate((obs['left_gripper'], obs['right_gripper']), 1).squeeze()
    for i in range(n_steps):
        viz.viz(env.physics)
        current_action = i / n_steps * action + (1 - i / n_steps) * initial
        print(current_action)
        time_step = env.step(current_action)
    viz.viz(env.physics)
    time_step = env.step(action)
    viz.viz(env.physics)
    return time_step.observation
