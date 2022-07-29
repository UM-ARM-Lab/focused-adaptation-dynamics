from time import perf_counter
from typing import Dict, Optional

import numpy as np
from dm_control import composer
from dm_control.rl.control import PhysicsError

import rospy
from dm_envs.mujoco_visualizer import MujocoVisualizer
from dm_envs.rope_task import RopeManipulation
from link_bot_data.color_from_kwargs import color_from_kwargs
from link_bot_pycommon.experiment_scenario import sample_delta_position
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
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

        self.last_action = None
        self.max_action_attempts = 100

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
        }
        init_state = self.get_state()
        self.execute_action(None, init_state, init_action)

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
        state['rope'] = state['rope'].reshape(75)
        state['left_gripper'] = state['left_gripper'].reshape(3)
        state['right_gripper'] = state['right_gripper'].reshape(3)
        return state

    def plot_action_rviz(self, state: Dict, action: Dict, **kwargs):
        pass

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate: bool,
                      stateless: Optional[bool] = False):
        for _ in range(self.max_action_attempts):
            # move in the same direction as the previous action with some probability
            repeat_probability = action_params['repeat_delta_gripper_motion_probability']
            if state.get('is_overstretched', False):
                left_gripper_delta_position = np.zeros(3, dtype=np.float)
                right_gripper_delta_position = np.zeros(3, dtype=np.float)
            elif not stateless and self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
                left_gripper_delta_position = self.last_action['left_gripper_delta_position']
                right_gripper_delta_position = self.last_action['right_gripper_delta_position']
            else:
                # Sample a new random action
                left_gripper_delta_position = sample_delta_position(action_params, action_rng)
                right_gripper_delta_position = sample_delta_position(action_params, action_rng)

            # Apply delta and check for out of bounds
            left_gripper_position = state['left_gripper'] + left_gripper_delta_position
            right_gripper_position = state['right_gripper'] + right_gripper_delta_position

            action = {
                'left_gripper_position':        left_gripper_position,
                'right_gripper_position':       right_gripper_position,
                'left_gripper_delta_position':  left_gripper_delta_position,
                'right_gripper_delta_position': right_gripper_delta_position,
            }

            if not validate or FloatingRopeScenario.is_action_valid(self, environment, state, action, action_params):
                self.last_action = action
                return action, (invalid := False)

        rospy.logwarn("Could not find a valid action, returning a zero action")
        zero_action = {
            'left_gripper_position':        state['left_gripper'],
            'right_gripper_position':       state['right_gripper'],
            'left_gripper_delta_position':  np.zeros(3, dtype=np.float),
            'right_gripper_delta_position': np.zeros(3, dtype=np.float),
        }
        return zero_action, (invalid := False)

    def execute_action(self, environment, state, action: Dict):
        # local controller with time and error based stopping conditions, as well as interpolation
        target_action_vec = np.concatenate((action['left_gripper_position'], action['right_gripper_position']))

        end_trial = False
        position_threshold = 0.001
        t0 = perf_counter()
        timeout = 30
        obs = self.env._observation_updater.get_observation()
        tmp_target_action_vec = np.concatenate((obs['left_gripper'], obs['right_gripper']), 1).squeeze()
        while True:
            self.viz.viz(self.env.physics)

            obs = self.env._observation_updater.get_observation()
            current_vec = np.concatenate((obs['left_gripper'], obs['right_gripper']), 1).squeeze()

            try:
                self.env.step(tmp_target_action_vec)
            except PhysicsError as e:
                print(e)
                end_trial = True
                break

            position_error = np.linalg.norm(current_vec - target_action_vec)
            positions_reached = position_error < position_threshold
            if positions_reached:
                break

            dt = perf_counter() - t0
            timeout_reached = dt > timeout
            if timeout_reached:
                break

            step_target_action_vec = np.clip((target_action_vec - tmp_target_action_vec), -0.01, 0.01)
            tmp_target_action_vec += step_target_action_vec

        self.viz.viz(self.env.physics)
        print(position_error)
        return end_trial

    def needs_reset(self, state: Dict, params: Dict):
        return False

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        self.env._random_state = env_rng
        self.env.reset()

    def reset_viz(self):
        super().reset_viz()

    def plot_state_rviz(self, state: Dict, **kwargs):
        super().plot_state_rviz(state, **kwargs)

        ns = kwargs.get("label", "")
        idx = kwargs.get("idx", 0)
        color_msg = color_from_kwargs(kwargs, 1.0, 0, 0.0)

        msg = MarkerArray()

        ig = marker_index_generator(idx)

        self.viz.viz(self.env.physics)

    def make_dm_task(self, params):
        return RopeManipulation(params)

    def __repr__(self):
        return "mj_floating_rope"
