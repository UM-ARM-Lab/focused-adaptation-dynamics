from typing import Dict

import mujoco
import numpy as np
from tqdm import tqdm

from dm_envs.mj_floating_rope_scenario import MjFloatingRopeScenario
from dm_envs.val_rope_task import ValRopeManipulation
from link_bot_pycommon.grid_utils_np import extent_to_env_shape, idx_to_point_3d_from_origin_point


class MjRopeValScenario(MjFloatingRopeScenario):

    def get_environment(self, params: Dict, **kwargs):
        env = super().get_environment(params)
        res = np.float32(params['res'])
        extent = np.array(params['extent'])
        shape = extent_to_env_shape(extent, res)

        geom_type = mujoco.mju_str2Type('geom')

        def in_collision(xyz):
            self.task.vgb.set_pose(self.env.physics, position=xyz)
            mujoco.mj_step1(self.env.physics.model.ptr, self.env.physics.data.ptr)
            for i, c in enumerate(self.env.physics.data.contact):
                geom1_name = mujoco.mj_id2name(self.env.physics.model.ptr, geom_type, c.geom1)
                geom2_name = mujoco.mj_id2name(self.env.physics.model.ptr, geom_type, c.geom2)
                if c.dist < 0 and (geom1_name == 'vgb_sphere/geom' or geom2_name == 'vgb_sphere/geom'):
                    # print(f"Contact at {xyz} between {geom1_name} and {geom2_name}, {c.dist=:.4f} {c.exclude=}")
                    return True
            return False

        vg = np.zeros(shape, dtype=np.float32)
        for row, col, channel in tqdm(np.ndindex(*shape), total=np.prod(shape)):
            xyz = idx_to_point_3d_from_origin_point(row, col, channel, res, env['origin_point'])
            if in_collision(xyz):
                vg[row, col, channel] = 1

        env.update({
            'res':    res,
            'extent': extent,
            'env':    env,
        })
        return env

    def get_state(self):
        state = super().get_state()
        state.update({
            'joint_names': np.array(self.task.actuated_joint_names),
        })
        return state

    def make_dm_task(self, params):
        return ValRopeManipulation(params)
