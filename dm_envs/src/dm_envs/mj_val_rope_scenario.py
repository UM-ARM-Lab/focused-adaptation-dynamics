from typing import Dict

import numpy as np
from dm_control import composer

from dm_envs.mj_floating_rope_scenario import MjFloatingRopeScenario
from dm_envs.val_rope_task import ValRopeManipulation


class MjValRopeScenario(MjFloatingRopeScenario):

    def __init__(self, params: Dict):
        super().__init__(params)

    def on_before_data_collection(self, params: Dict):
        self.task = self.make_dm_task(params)
        # we don't want episode termination to be decided by dm_control, we do that ourselves elsewhere
        self.env = composer.Environment(self.task, time_limit=9999, random_state=0)
        self.env.reset()
        self.action_spec = self.env.action_spec()

        extent = np.array(params['extent']).reshape([3, 2])
        cx = extent[0].mean()
        cy = extent[1].mean()
        min_z = extent[2, 0]
        left_gripper_position = np.array([cx, cy + 0.25, min_z + 0.6])
        right_gripper_position = np.array([cx, cy - 0.25, min_z + 0.6])
        init_action = {
            'left_gripper_position':  left_gripper_position,
            'right_gripper_position': right_gripper_position,
            'left_gripper_quat':      params['left_gripper_quat'],
            'right_gripper_quat':     params['right_gripper_quat'],
        }
        init_state = self.get_state()
        self.execute_action(None, init_state, init_action)

    def get_state(self):
        state = super().get_state()
        state.update({
            'joint_names': np.array(self.task.actuated_joint_names),
        })
        return state

    def execute_action(self, environment, state, action: Dict):
        end_trial = False
        return end_trial



        from tqdm import tqdm
        from time import perf_counter
        from rviz_voxelgrid_visuals_msgs.msg import VoxelgridStamped

        pub = get_connected_publisher('/occupancy', VoxelgridStamped, queue_size=1)

        geom_type = mujoco.mju_str2Type('geom')

        def in_collision(xyz):
            self.vgb.set_pose(physics, position=xyz)
            mujoco.mj_step1(physics.model.ptr, physics.data.ptr)
            for i, c in enumerate(physics.data.contact):
                geom1_name = mujoco.mj_id2name(physics.model.ptr, geom_type, c.geom1)
                geom2_name = mujoco.mj_id2name(physics.model.ptr, geom_type, c.geom2)
                if c.dist < 0 and (geom1_name == 'vgb_sphere/geom' or geom2_name == 'vgb_sphere/geom'):
                    # print(f"Contact at {xyz} between {geom1_name} and {geom2_name}, {c.dist=:.4f} {c.exclude=}")
                    return True
            return False

        t0 = perf_counter()
        res = 0.02
        rows = 60
        columns = 50
        channels = 55
        vg = np.zeros([rows, columns, channels], dtype=np.float32)
        origin_point = np.array([0, -0.5, 0])

        for row, col, channel in tqdm(np.ndindex(rows, columns, channels), total=rows * columns * channels):
            xyz = idx_to_point_3d_from_origin_point(row, col, channel, res, origin_point)
            if in_collision(xyz):
                vg[row, col, channel] = 1
        print(perf_counter() - t0)

        pub.publish(vox_to_voxelgrid_stamped(vg, scale=res, frame='world'))

    def make_dm_task(self, params):
        return ValRopeManipulation(params)

    def __repr__(self):
        return "mj_val_rope"
