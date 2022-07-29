from dm_control import composer

from link_bot_pycommon.base_services import BaseServices


class MujocoServices(BaseServices):

    def __init__(self):
        super().__init__()

    def setup_env(self, *args, **kwargs):
        pass


def my_step(task, env: composer.Environment, action, n_steps=1):
    initial = task.current_action_vec(env.physics)
    for i in range(n_steps):
        task.viz(env.physics)
        current_action = i / n_steps * action + (1 - i / n_steps) * initial
        env.step(current_action)

    time_step = env.step(action)
    task.viz(env.physics)

    return time_step.observation
