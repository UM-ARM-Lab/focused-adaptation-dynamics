from typing import Dict, Optional
from copy import deepcopy
from typing import Dict, Optional, List

import numpy as np
import tensorflow as tf
from matplotlib import colors

import ros_numpy
import rospy
from arc_utilities.algorithms import nested_dict_update
from arc_utilities.listener import Listener
from arc_utilities.marker_utils import scale_marker_array
from augmentation.aug_opt import compute_moved_mask
from augmentation.aug_opt_utils import get_local_frame
from geometry_msgs.msg import Point, Vector3
from jsk_recognition_msgs.msg import BoundingBox
from learn_invariance.transform_link_states import transform_link_states
from dm_envs.softgym_services import SoftGymServices
from link_bot_data.base_collect_dynamics_data import collect_trajectory
from link_bot_data.coerce_types import coerce_types
from link_bot_data.dataset_utils import get_maybe_predicted, in_maybe_predicted, add_predicted, add_predicted_cond
from link_bot_data.rviz_arrow import rviz_arrow
from link_bot_gazebo.gazebo_services import gz_scope, restore_gazebo, GazeboServices
from link_bot_gazebo.gazebo_utils import get_gazebo_kinect_pose
from link_bot_gazebo.position_3d import Position3D
from link_bot_pycommon.bbox_marker_utils import make_box_marker_from_extents
from link_bot_pycommon.bbox_visualization import viz_action_sample_bbox
from link_bot_pycommon.debugging_utils import debug_viz_batch_indices
from link_bot_pycommon.experiment_scenario import get_action_sample_extent, is_out_of_bounds, sample_delta_position
from link_bot_pycommon.get_link_states import GetLinkStates
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from link_bot_pycommon.grid_utils_np import extent_to_env_shape, extent_res_to_origin_point
from link_bot_pycommon.lazy import Lazy
from link_bot_pycommon.make_rope_markers import make_gripper_marker, make_rope_marker
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.matplotlib_utils import adjust_lightness_msg
from link_bot_pycommon.moveit_planning_scene_mixin import MoveitPlanningSceneScenarioMixin
from link_bot_pycommon.pycommon import default_if_none
from link_bot_pycommon.ros_pycommon import publish_color_image, publish_depth_image, get_camera_params
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.geometry_np import homogeneous
from moonshine.geometry_tf import xyzrpy_to_matrices, transform_points_3d, densify_points
from moonshine.grid_utils_tf import batch_center_res_shape_to_origin_point, dist_to_bbox
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from peter_msgs.srv import *
from rosgraph.names import ns_join
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, EmptyRequest
from visualization_msgs.msg import MarkerArray, Marker

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
from autolab_core import YamlConfig

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
        self.state_color_viz_pub = rospy.Publisher("state_color_viz", Image, queue_size=10, latch=True)
        self.state_depth_viz_pub = rospy.Publisher("state_depth_viz", Image, queue_size=10, latch=True)
        self.last_action = None
        self.pos3d = Position3D()
        self._params = params
        self.max_action_attempts = 100
        self.robot_reset_rng = np.random.RandomState(0)
        self.control_volumes = []
        self.target_volumes = []
        self._make_softgym_env()
        self.service_provider = SoftGymServices()
        self.service_provider.set_scene(self._scene)

    def _make_softgym_env(self):
        softgym_env_name = "PourWater"
        env_kwargs = env_arg_dict[softgym_env_name]

        # Generate and save the initial states for running this environment for the first time
        env_kwargs['use_cached_states'] = False
        env_kwargs['save_cached_states'] = False
        env_kwargs['num_variations'] = 1
        env_kwargs['render'] = 1  # True
        env_kwargs["action_repeat"] = 2
        env_kwargs['headless'] = 1 #not self._params['gui']
        self._save_cfg = self._params["save_cfg"]

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

    def get_environment(self, params: Dict, **kwargs):
        res = params["res"]

        voxel_grid_env = get_environment_for_extents_3d(extent=params['extent'],
                                                        res=res,
                                                        frame='map',
                                                        service_provider=self.service_provider,
                                                        excluded_models=[])

        env = {}
        env.update({k: np.array(v).astype(np.float32) for k, v in voxel_grid_env.items()})
        return env

    def hard_reset(self):
        raise NotImplementedError

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
        angle_traj = np.linspace(curr_angle, goal_angle, int(self._params["controller_max_horizon"]*0.5))
        pos_traj = np.vstack([np.linspace(curr_pos[dim], goal_pos[dim], int(self._params["controller_max_horizon"]*0.5)) for dim in range(curr_pos.shape[-1])]).T
        for i in range(self._params["controller_max_horizon"]):
            curr_pos = curr_state["controlled_container_pos"].flatten()
            curr_angle = curr_state["controlled_container_angle"].flatten()
            traj_idx = min(i, len(angle_traj)-1)
            target_pos = pos_traj[traj_idx]
            target_angle = angle_traj[traj_idx]
            pos_error = target_pos - curr_pos
            pos_control = self._params["k_pos"] * (pos_error)
            angle_error = target_angle - curr_angle
            angle_control = self._params["k_angle"] * (angle_error)

            vector_action = np.hstack([pos_control, angle_control])
            self._saved_data = self._scene.step(vector_action, record_continuous_video=self._save_frames,
                                                img_size=self._save_cfg["img_size"])
            _, _, _, info = self._saved_data
            curr_state = self.get_state()
            if self._save_frames:
                self.frames.extend(info['flex_env_recorded_frames'])
        return False 

    def dynamics_dataset_metadata(self):
        metadata = ScenarioWithVisualization.dynamics_dataset_metadata(self)
        return metadata

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
                random_angle = np.array(action_rng.uniform(low=self._params["action_range"]["angle"][0],
                                                  high=self._params["action_range"]["angle"][1]), dtype=np.float32)
                action = {"controlled_container_target_pos":   current_controlled_container_pos,
                          "controlled_container_target_angle": random_angle.reshape(-1,1)}
            else:
                random_x = action_rng.uniform(low=self._params["action_range"]["x"][0],
                                              high=self._params["action_range"]["x"][1])
                random_z = action_rng.uniform(low=self._params["action_range"]["z"][0],
                                              high=self._params["action_range"]["z"][1])
                action = {"controlled_container_target_pos":   np.array([random_x, random_z], dtype=np.float32),
                          "controlled_container_target_angle": np.array(0, dtype=np.float32).reshape(-1,1)}
            if self.is_action_valid(environment, state, action, action_params):
                return action, False
        if self.is_action_valid(environment, state, action, action_params):
            return action, False
        return None, True

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        curr_pos = state["controlled_container_pos"]
        target_pos = action["controlled_container_target_pos"]
        curr_height = curr_pos[1]
        max_dist = 0.3
        if np.linalg.norm(curr_pos-target_pos) > max_dist:
            return False
        min_height_for_pour = 0.25
        is_pour = action["controlled_container_target_angle"] >= 0.001
        if is_pour and curr_height < min_height_for_pour:
            return False
        return True

    def actions_cost(self, states: List[Dict], actions: List[Dict], action_params: Dict):
        return 0

    def actions_cost_torch(self, states: List[Dict], actions: List[Dict], action_params: Dict):
        return 0

    @staticmethod
    def robot_name():
        return "control_box"

    def reset(self):
        self._scene.reset()
        null_action = np.zeros(3,)
        self._saved_data = self._scene.step(null_action, record_continuous_video=self._save_frames,
                                            img_size=self._save_cfg["img_size"])

    def randomize_environment(self, env_rng, params: Dict):
        self.reset()

    @staticmethod
    def put_state_local_frame(state: Dict):
        return NotImplementedError

    @staticmethod
    def add_action_noise(action: Dict, noise_rng: np.random.RandomState):
        raise NotImplementedError

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_pos = action["controlled_container_target_pos"]
        target_angle = action["controlled_container_target_angle"]
        current_pos = state["controlled_container_pos"]
        current_angle = state["controlled_container_angle"]
        delta_pos = target_pos - current_pos
        delta_angle = target_angle - current_angle
        return {
            'delta_pos':   delta_pos,
            'delta_angle': delta_angle
        }

    @staticmethod
    def apply_local_action_at_state(state, local_action):
        delta_pos = state['delta_pos']
        delta_angle = state['delta_angle']

        local_action = {"controlled_container_target_pos":   state["current_controlled_container_pos"] + delta_pos,
                        "controlled_container_target_angle": (
                                    state["current_controlled_container_angle"] + delta_angle)}
        return local_action

    @staticmethod
    def put_state_local_frame_torch(state: Dict):
        target_pos = state["target_container_pos"]
        current_pos = state["controlled_container_pos"]
        current_angle = state["controlled_container_angle"]
        delta_pos = target_pos - current_pos

        local_dict = {
            'controlled_container_pos_local':   delta_pos,
            'controlled_container_angle_local': current_angle,
        }
        local_dict["target_volume"] = state["target_volume"]
        local_dict["control_volume"] = state["control_volume"]
        local_dict["target_container_pos"] = 0 * state["target_container_pos"]
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
        # cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
        # self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
        # self._get_current_water_height(), in_poured_glass, in_control_glass])
        state = {"controlled_container_pos":   state_vector[:2],
                 "controlled_container_angle": state_vector[2].reshape(-1,1),
                 "target_container_pos":       np.array([state_vector[6], 0]),  # need to check this one
                 "control_volume":             state_vector[-1].reshape(-1,1),
                 "target_volume":              state_vector[-2].reshape(-1,1)}
        #self.control_volumes.append(state_vector[-1])
        #self.target_volumes.append(state_vector[-2])
        return state

    def distance_to_goal(self, state: Dict, goal: Dict, use_torch=False):
        goal_target_volume = goal["goal_target_volume"]
        return abs(state["target_volume"] - goal_target_volume)

    def goal_state_to_goal(self, goal_state: Dict, goal_type: str):
        raise NotImplementedError()

    def plot_goal_rviz(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        if actually_at_goal:
            r = 0.4
            g = 0.8
            b = 0.4
            a = 0.6
        else:
            r = 0.5
            g = 0.3
            b = 0.8
            a = 0.6
        goal_marker_msg = MarkerArray()
        self.state_viz_pub.publish(goal_marker_msg)

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        raise NotImplementedError

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        raise NotImplementedError

    def plot_environment_rviz(self, environment: Dict, **kwargs):
        ScenarioWithVisualization.plot_environment_rviz(self, environment, **kwargs)

    def plot_state_rviz(self, state: Dict, **kwargs):
        raise NotImplementedError

    def plot_action_rviz(self, state: Dict, action: Dict, label: str = 'action', **kwargs):
        state_action = {}
        raise NotImplementedError

    def simple_name(self):
        return "watering"

    def __repr__(self):
        return self.simple_name()

