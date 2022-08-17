from typing import Dict, Optional, List

import numpy as np
import torch

from dm_envs.softgym_services import SoftGymServices
from link_bot_pycommon.experiment_scenario import MockRobot
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from link_bot_pycommon.grid_utils_np import extent_to_env_shape, extent_res_to_origin_point

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif

from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization

control_box_name = "control_box"


class WaterSimScenario(ScenarioWithVisualization):
    """
    Representation:
    state: target/control pos, target/control volume
    goal: target_volume, tolerance
    action: goal x,z of control. theta, which can be 0.
    """

    def __init__(self, params: Optional[dict] = None):
        ScenarioWithVisualization.__init__(self, params)
        self.max_action_attempts = 100
        self.robot_reset_rng = np.random.RandomState(0)
        self.control_volumes = []
        self.target_volumes = []
        if self.params.get("run_flex", False):
            self._make_softgym_env()
        self.service_provider = SoftGymServices()
        if self.params.get("run_flex", False):
            self.service_provider.set_scene(self._scene)
        self.robot = MockRobot()
        self.robot.jacobian_follower = None

    def _make_softgym_env(self):
        softgym_env_name = "PourWater"
        env_kwargs = env_arg_dict[softgym_env_name]

        # Generate and save the initial states for running this environment for the first time
        env_kwargs['use_cached_states'] = False
        env_kwargs['save_cached_states'] = False
        env_kwargs['num_variations'] = 1
        env_kwargs['render'] = True
        env_kwargs["action_repeat"] = 2
        env_kwargs['headless'] = not self.params['gui']
        self._save_cfg = self.params["save_cfg"]

        if not env_kwargs['use_cached_states']:
            print('Waiting to generate environment variations. May take 1 minute for each variation...')
        self._scene = normalize(SOFTGYM_ENVS[softgym_env_name](**env_kwargs))
        self._save_frames = self._save_cfg["save_frames"]
        if self._save_frames:
            self.frames = [self._scene.get_image(self._save_cfg["img_size"], self._save_cfg["img_size"])]

    def needs_reset(self, state: Dict, params: Dict):
        if state["control_volume"].item() < 0.5:
            return True
        total_water_in_containers = state["control_volume"].item() + state["target_volume"].item()
        if total_water_in_containers < 0.5:
            return True
        return False

    def classifier_distance(self, s1, s2):
        """ this is not the distance metric used in planning """
        container_dist = np.linalg.norm(s1["controlled_container_pos"] - s2["controlled_container_pos"], axis=1)
        return container_dist

    def get_environment(self, params: Dict, **kwargs):
        res = params["res"]

        voxel_grid_env = get_environment_for_extents_3d(extent=params['extent'],
                                                        res=res,
                                                        frame='map',
                                                        service_provider=self.service_provider,
                                                        excluded_models=[])

        env = {}
        env.update({k: np.array(v).astype(np.float32) for k, v in voxel_grid_env.items()})
        env["origin_point"] = extent_res_to_origin_point(extent=params['extent'], res=res)
        env["res"] = res
        return env

    def on_before_get_state_or_execute_action(self):
        self.on_before_action()

    def on_before_action(self):
        pass

    def on_before_data_collection(self, params: Dict):
        self.reset()

    def on_after_data_collection(self, params: Dict):
        if self._save_frames:
            save_name = "test.gif"
            save_numpy_as_gif(np.array(self.frames), save_name)
            print("Saved to", save_name)

    def execute_action(self, environment, state, action: Dict, **kwargs):
        goal_pos = action["controlled_container_target_pos"].flatten()
        goal_angle = action["controlled_container_target_angle"].flatten()
        curr_state = state
        curr_pos = curr_state["controlled_container_pos"].flatten()
        curr_angle = curr_state["controlled_container_angle"].flatten()
        angle_traj = np.linspace(curr_angle, goal_angle, int(self.params["controller_max_horizon"] * 0.5))
        pos_traj = np.vstack(
            [np.linspace(curr_pos[dim], goal_pos[dim], int(self.params["controller_max_horizon"] * 0.5)) for dim in
             range(curr_pos.shape[-1])]).T
        for i in range(self.params["controller_max_horizon"]):
            curr_pos = curr_state["controlled_container_pos"].flatten()
            curr_angle = curr_state["controlled_container_angle"].flatten()
            traj_idx = min(i, len(angle_traj) - 1)
            target_pos = pos_traj[traj_idx]
            target_angle = angle_traj[traj_idx]
            pos_error = target_pos - curr_pos
            pos_control = self.params["k_pos"] * (pos_error)
            angle_error = target_angle - curr_angle
            angle_control = self.params["k_angle"] * (angle_error)

            vector_action = np.hstack([pos_control, angle_control])
            self._saved_data = self._scene.step(vector_action, record_continuous_video=self._save_frames,
                                                img_size=self._save_cfg["img_size"])
            _, _, _, info = self._saved_data
            curr_state = self.get_state()
            if self._save_frames:
                self.frames.extend(info['flex_env_recorded_frames'])
        return False

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate: bool,
                      stateless: Optional[bool] = False):
        current_controlled_container_pos = state["controlled_container_pos"]
        for _ in range(self.max_action_attempts):
            is_pour = action_rng.randint(2)
            if is_pour:
                random_angle = np.array([action_rng.uniform(low=self.params["action_range"]["angle"][0],
                                                           high=self.params["action_range"]["angle"][1])],
                                        dtype=np.float32)
                action = {"controlled_container_target_pos":   current_controlled_container_pos,
                          "controlled_container_target_angle": random_angle}
            else:
                random_x = action_rng.uniform(low=self.params["action_range"]["x"][0],
                                              high=self.params["action_range"]["x"][1])
                random_z = action_rng.uniform(low=self.params["action_range"]["z"][0],
                                              high=self.params["action_range"]["z"][1])
                action = {"controlled_container_target_pos":   np.array([random_x, random_z], dtype=np.float32),
                          "controlled_container_target_angle": np.array([0], dtype=np.float32)}
            if validate and self.is_action_valid(environment, state, action, action_params):
                return action, False
        return None, True

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        curr_pos = state["controlled_container_pos"]
        target_pos = action["controlled_container_target_pos"]
        curr_height = curr_pos[1]
        max_dist = 0.3
        if np.linalg.norm(curr_pos - target_pos) > max_dist:
            return False
        min_height_for_pour = 0.25
        is_pour = action["controlled_container_target_angle"] >= 0.001
        if is_pour and curr_height < min_height_for_pour:
            return False
        return True

    def actions_cost(self, states: List[Dict], actions: List[Dict], action_params: Dict):
        return 1

    def actions_cost_torch(self, states: List[Dict], actions: List[Dict], action_params: Dict):
        return 1

    @staticmethod
    def robot_name():
        return "control_box"

    @staticmethod
    def local_environment_center_differentiable_torch(state):
        pos_xz = state["target_container_pos"].unsqueeze(-1)
        local_center = torch.cat((pos_xz[:, 0], torch.zeros_like(pos_xz[:, 0]), pos_xz[:, 1]), dim=1)
        if len(local_center.shape) == 0:
            return local_center.reshape(1, -1)
        return local_center

    def reset(self):
        self._scene.reset()
        null_action = np.zeros(3, )
        self._saved_data = self._scene.step(null_action, record_continuous_video=self._save_frames,
                                            img_size=self._save_cfg["img_size"])

    def randomize_environment(self, env_rng, params: Dict):
        self.reset()


    @staticmethod
    def put_state_robot_frame(state: Dict):
        # Assumes everything is in robot frame already
        return state

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_pos = action["controlled_container_target_pos"]
        target_angle = action["controlled_container_target_angle"]
        target_angle = _fix_extremes_1d_data(target_angle)

        current_pos = state["controlled_container_pos"]
        current_angle = state["controlled_container_angle"]

        if len(current_angle.shape) == 3:  # fix dataset:
            current_angle = current_angle.squeeze(-1)
        delta_pos = target_pos - current_pos
        delta_angle = target_angle - current_angle
        return {
            'delta_pos':   delta_pos.float(),
            'delta_angle': delta_angle.float()
        }

    @staticmethod
    def apply_local_action_at_state(state, local_action):
        delta_pos = state['delta_pos']
        delta_angle = state['delta_angle']
        curr_angle = _fix_extremes_1d_data(state["current_controlled_container_angle"])

        local_action = {"controlled_container_target_pos":   state["current_controlled_container_pos"] + delta_pos,
                        "controlled_container_target_angle": (
                                curr_angle + delta_angle)}
        return local_action

    @staticmethod
    def put_state_local_frame_torch(state: Dict):
        target_pos = state["target_container_pos"]
        current_pos = state["controlled_container_pos"]
        current_angle = state["controlled_container_angle"]
        delta_pos = target_pos - current_pos

        local_dict = {
            'controlled_container_pos_local':   delta_pos.float(),
            'controlled_container_angle_local': _squeeze_if_3d(current_angle).float(),
        }
        local_dict["target_volume"] = _squeeze_if_3d(state["target_volume"]).float()
        local_dict["control_volume"] = _squeeze_if_3d(state["control_volume"]).float()
        local_dict["target_container_pos"] = 0 * state["target_container_pos"].float()
        return local_dict

    @staticmethod
    def integrate_dynamics(s_t: Dict, delta_s_t: Dict):
        integrated_dynamics = {}
        for key in s_t.keys():
            if s_t[key].shape == delta_s_t[key].shape:
                integrated_dynamics[key] = s_t[key] + delta_s_t[key]
            else:
                integrated_dynamics[key] = s_t[key].reshape(delta_s_t[key].shape) + delta_s_t[key]
        return integrated_dynamics

    def get_state(self):
        state_vector = self._saved_data[0][0]
        state = {"controlled_container_pos":   state_vector[:2],
                 "controlled_container_angle": np.array([state_vector[2]],dtype=np.float32),
                 "target_container_pos":       np.array([state_vector[6], 0]),  # need to check this one
                 "control_volume":             np.array([state_vector[-1]], dtype=np.float32),
                 "target_volume":              np.array([state_vector[-2]], dtype=np.float32)}
        return state

    def distance_to_goal(self, state: Dict, goal: Dict, use_torch=False):
        goal_target_volume = goal["goal_target_volume"]
        return abs(state["target_volume"] - goal_target_volume)

    def simple_name(self):
        return "watering"

    def __repr__(self):
        return self.simple_name()

#Specific to water scenario. A lot of functions seem to assume 2D data and then it gets converted
def _fix_extremes_1d_data(data):
    if len(data.shape) == 1:
        data = data.unsqueeze(-1)
        return data
    if len(data.shape) == 4:
        data = data.squeeze(-1)
        return data
    return data

def _squeeze_if_3d(data):
    if len(data.shape) == 3:
        data = data.squeeze(-1)
    return data
   

