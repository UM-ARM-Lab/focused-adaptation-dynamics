from typing import Dict, Optional

import numpy as np
from dm_control import composer

import rospy
from dm_envs.mujoco_visualizer import MujocoVisualizer
from dm_envs.rope_task import RopeManipulation
from link_bot_data.color_from_kwargs import color_from_kwargs
from link_bot_data.visualization_common import make_delete_markerarray
from link_bot_pycommon.bbox_visualization import viz_action_sample_bbox
from link_bot_pycommon.experiment_scenario import get_action_sample_extent, is_out_of_bounds
from link_bot_pycommon.grid_utils_np import extent_to_env_shape
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from visualization_msgs.msg import MarkerArray


class MjFloatingRopeScenario(ScenarioWithVisualization):

    def __init__(self, params: Dict):
        super().__init__(params)
        self.task = None
        self.env = None
        self.action_spec = None

        self.viz = MujocoVisualizer()

    def on_before_data_collection(self, params: Dict):
        self.task = self.make_dm_task(params)
        # we don't want episode termination to be decided by dm_control, we do that ourselves elsewhere
        self.env = composer.Environment(self.task, time_limit=9999, random_state=0)
        self.env.reset()
        self.action_spec = self.env.action_spec()

    def get_environment(self, params: Dict, **kwargs):
        # not the mujoco "env", this means the static obstacles and workspaces geometry
        res = np.float32(params['res'])
        extent = np.array(params['extent'])
        origin_point = extent[[0, 2, 4]]
        shape = extent_to_env_shape(extent, res)
        empty_env = np.zeros(shape, np.float32)

        return {
            'res':          res,
            'extent':       extent,
            'env':          empty_env,
            'origin_point': origin_point,
        }

    def get_state(self):
        state = self.env._observation_updater.get_observation()
        joint_names = [n.replace(f'{ARM_NAME}/', '') for n in self.task.joint_names]
        state['joint_names'] = np.array(joint_names)

        return state

    def plot_action_rviz(self, state: Dict, action: Dict, **kwargs):
        pass

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate, stateless: Optional[bool] = False):
        viz_action_sample_bbox(self.gripper_bbox_pub, get_action_sample_extent(action_params))

        start_gripper_position = get_tcp_pos(state)
        action_dict = {
            'gripper_position': start_gripper_position,
        }

        # first check if any objects are wayyy to far
        num_objs = state['num_objs'][0]
        for i in range(num_objs):
            obj_position = state[f'obj{i}/position'][0]
            out_of_bounds = is_out_of_bounds(obj_position, action_params['extent'])
            if out_of_bounds:
                return action_dict, (invalid := True)  # this will cause the current trajectory to be thrown out

        for _ in range(self.max_action_attempts):
            repeat_probability = action_params['repeat_delta_gripper_motion_probability']
            if self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
                gripper_delta_position = self.last_action['gripper_delta_position']
            else:
                gripper_delta_position = sample_delta_xy(action_params, action_rng)

            gripper_position = start_gripper_position + gripper_delta_position
            gripper_position[2] = ACTION_Z

            self.tf.send_transform(gripper_position, [0, 0, 0, 1], 'world', 'sample_action_gripper_position')

            out_of_bounds = is_out_of_bounds(gripper_position, action_params['gripper_action_sample_extent'])
            if out_of_bounds and validate:
                self.last_action = None
                continue

            action = {
                'gripper_position':       gripper_position,
                'gripper_delta_position': gripper_delta_position,
            }

            self.last_action = action
            return action, (invalid := False)

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action_dict, (invalid := False)

    def execute_action(self, environment, state, action: Dict):
        target_cartesian_position = action['gripper_position']

        # we picked a new end effector pose, now solve IK to turn that into a joint configuration
        success, target_joint_position = self.task.solve_position_ik(self.env.physics, target_cartesian_position)
        if not success:
            rospy.logwarn("failed to solve IK!")
            return (end_trial := True)

        current_position = get_joint_position(state)
        kP = 10.0

        max_substeps = 50
        for substeps in range(max_substeps):
            # p-control to achieve joint positions using the lower level velocity controller
            velocity_cmd = yaw_diff(target_joint_position, current_position) * kP
            self.env.step(velocity_cmd)
            state = self.get_state()
            # self.plot_state_rviz(state, label='actual')

            current_position = get_joint_position(state)
            max_error = max(yaw_diff(target_joint_position, current_position))
            max_vel = max(abs(get_joint_velocities(state)))
            reached = max_error < 0.01
            stopped = max_vel < 0.002
            if reached and stopped:
                break

        return (end_trial := False)

    def needs_reset(self, state: Dict, params: Dict):
        return False

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        self.env._random_state = env_rng
        self.env.reset()

    def reset_viz(self):
        super().reset_viz()
        m = make_delete_markerarray(ns='viz_aug')
        self.viz_aug_pub.publish(m)

    def plot_state_rviz(self, state: Dict, **kwargs):
        super().plot_state_rviz(state, **kwargs)

        ns = kwargs.get("label", "")
        idx = kwargs.get("idx", 0)
        color_msg = color_from_kwargs(kwargs, 1.0, 0, 0.0)

        msg = MarkerArray()

        ig = marker_index_generator(idx)

        # self.viz.viz(physics)

    def make_dm_task(self, params):
        return RopeManipulation(params)

    def __repr__(self):
        return "mj_floating_rope"
