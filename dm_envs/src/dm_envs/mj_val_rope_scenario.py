from time import perf_counter
from typing import Dict

import mujoco
import numpy as np
from tqdm import tqdm

from dm_envs.mj_floating_rope_scenario import MjFloatingRopeScenario, interp_to
from dm_envs.val_rope_task import ValRopeManipulation
from link_bot_pycommon.grid_utils_np import extent_to_env_shape, idx_to_point_3d_from_origin_point


class MjValRopeScenario(MjFloatingRopeScenario):

    def __init__(self, params: Dict):
        super().__init__(params)

    def on_before_data_collection(self, params: Dict):
        super().setup_task_and_env(params)

        self.task.grasp_rope(self.env.physics)

        extent = np.array(params['extent']).reshape([3, 2])
        cx = extent[0].mean()
        cy = extent[1].mean()
        min_z = extent[2, 0]
        left_gripper_position = np.array([cx, cy + 0.25, min_z + 0.8])
        right_gripper_position = np.array([cx, cy - 0.25, min_z + 0.8])
        init_action = {
            'left_gripper_position':  left_gripper_position,
            'right_gripper_position': right_gripper_position,
            'left_gripper_quat':      params['left_gripper_quat'],
            'right_gripper_quat':     params['right_gripper_quat'],
        }

        init_state = self.get_state()
        self.execute_action(None, init_state, init_action)

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

    def execute_action(self, environment, state, action: Dict):
        end_trial = False

        # local controller with time and error based stopping conditions, as well as interpolation
        target_action_vec = np.concatenate((action['left_gripper_position'], action['left_gripper_quat'],
                                            action['right_gripper_position'], action['right_gripper_quat']))

        end_trial = False
        position_threshold = 0.001
        desired_speed = 0.1  # m/s
        time_fudge_factor = 2
        d_per_step = desired_speed * self.env.control_timestep()
        expected_distance = np.mean([np.linalg.norm(action['left_gripper_position'] - state['left_gripper']),
                                     np.linalg.norm(action['right_gripper_position'] - state['right_gripper'])])
        timeout = expected_distance / desired_speed * time_fudge_factor
        tmp_target_action_vec = self.task.current_action_vec(self.env.physics)

        t0 = perf_counter()
        while True:
            tmp_target_action_vec
            joint_action_vec = self.task.solve_ik(target_pos=[])
            # _, qdes = task.solve_ik(target_pos=[0, task.rope.length_m / 2, 0.05],
            #                         target_quat=quaternion_from_euler(0, -np.pi, 0),
            #                         site_name='val/left_tool')
            self.env.step(joint_action_vec)

            current_vec = self.task.current_action_vec(self.env.physics)

            position_error = np.linalg.norm(current_vec - target_action_vec)
            positions_reached = position_error < position_threshold
            if positions_reached:
                break

            dt = perf_counter() - t0
            timeout_reached = dt > timeout
            if timeout_reached:
                break

            tmp_target_action_vec = interp_to(target_action_vec, current_vec, d_per_step)

        return end_trial

        return end_trial

    def make_dm_task(self, params):
        return ValRopeManipulation(params)

    def __repr__(self):
        return "mj_val_rope"
