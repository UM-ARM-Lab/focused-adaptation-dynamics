from dm_control import composer

from link_bot_pycommon.base_services import BaseServices


class MujocoServices(BaseServices):

    def __init__(self):
        super().__init__()

    def setup_env(self, *args, **kwargs):
        pass


def my_step(task, env: composer.Environment, action, n_seconds):
    n_steps = int(n_seconds / task.control_timestep)
    initial = task.current_action_vec(env.physics)
    for i in range(n_steps):
        current_action = i / n_steps * action + (1 - i / n_steps) * initial
        env.step(current_action)

    time_step = env.step(action)

    return time_step.observation
