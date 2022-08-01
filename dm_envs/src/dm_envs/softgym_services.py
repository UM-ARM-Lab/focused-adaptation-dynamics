from dm_control import composer

from link_bot_pycommon.base_services import BaseServices


class SoftGymServices(BaseServices):

    def __init__(self):
        super().__init__()

    def setup_env(self, *args, **kwargs):
        pass

