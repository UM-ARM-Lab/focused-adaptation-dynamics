from typing import Dict

import mujoco
import numpy as np
from tqdm import tqdm

import dm_envs.abstract_follow_trajectory
from arm_robots.hdt_michigan import Val
from dm_envs.mj_floating_rope_scenario import MjFloatingRopeScenario
from dm_envs.val_rope_task import ValRopeManipulation
from link_bot_pycommon.grid_utils_np import extent_to_env_shape, idx_to_point_3d_from_origin_point
from tf import transformations


class MjValRopeScenario(MjFloatingRopeScenario):

    def __init__(self, params: Dict):
        super().__init__(params)

        self.val_planner = Val()
        self.val_planner.set_execute(False)
        self.val_planner.store_tool_orientations({
            # Here we use tf.transformations because it uses [x,y,z,w] convention which is what arm_robots uses
            'left_tool':  transformations.quaternion_from_euler(-1.779, -1.043, -2.0),
            'right_tool': transformations.quaternion_from_euler(np.pi, -1.408, 0.9),
        })

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

        for i in range(100):
            self.env.step(None)

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
        for row, col, channel in np.ndindex(*shape):
            xyz = idx_to_point_3d_from_origin_point(row, col, channel, res, env['origin_point'])
            if in_collision(xyz):
                vg[row, col, channel] = 1

        env['env'] = vg
        return env

    def get_state(self):
        state = super().get_state()
        state.update({
            'joint_names': np.array(self.task.actuated_joint_names),
        })
        return state

    def execute_action(self, environment, state, action: Dict):
        end_trial = False

        tool_names = [self.val_planner.left_tool_name, self.val_planner.right_tool_name]
        scene = self.task.get_planning_scene_msg(self.env.physics)
        left_gripper_point = action['left_gripper_position']
        right_gripper_point = action['right_gripper_position']
        grippers = [[left_gripper_point], [right_gripper_point]]
        result = self.val_planner.follow_jacobian_to_position_from_scene_and_state(group_name="both_arms",
                                                                                   scene_msg=scene,
                                                                                   joint_state=scene.robot_state.joint_state,
                                                                                   tool_names=tool_names,
                                                                                   points=grippers,
                                                                                   vel_scaling=1)
        dm_envs.abstract_follow_trajectory.follow_trajectory(self.env, result.planning_result.plan.joint_trajectory)

        return end_trial

    def make_dm_task(self, params):
        return ValRopeManipulation(params)

    def __repr__(self):
        return "mj_val_rope"
