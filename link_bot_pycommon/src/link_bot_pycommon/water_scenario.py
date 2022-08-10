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
        self._make_softgym_env()

    def _make_softgym_env(self):
        softgym_env_name = "PourWater"
        env_kwargs = env_arg_dict[softgym_env_name]

        # Generate and save the initial states for running this environment for the first time
        env_kwargs['use_cached_states'] = False
        env_kwargs['save_cached_states'] = False
        env_kwargs['num_variations'] = 1
        env_kwargs['render'] = 1  # True
        env_kwargs["action_repeat"] = 2
        env_kwargs['headless'] = not self._params['gui']
        self._n_envs = 1
        self._env_idxs = [0]
        self._save_cfg = self._params["save_cfg"]

        if not env_kwargs['use_cached_states']:
            print('Waiting to generate environment variations. May take 1 minute for each variation...')
        self._scene = normalize(SOFTGYM_ENVS[softgym_env_name](**env_kwargs))
        self._save_frames = self._save_cfg["save_frames"]
        if self._save_frames:
            self.frames = [self._scene.get_image(self._save_cfg["img_size"], self._save_cfg["img_size"])]

    def needs_reset(self, state: Dict, params: Dict):
        total_water_in_containers = state["control_volume"] + state["target_volume"]
        if total_water_in_containers < 0.5:
            return True
        return False

    def get_environment(self, params: Dict, **kwargs):
        env = {"env": np.zeros((40, 40, 40))}
        return env

    def hard_reset(self):
        raise NotImplementedError

    def on_before_get_state_or_execute_action(self):
        self.on_before_action()

    def on_before_action(self):
        pass

    def on_before_data_collection(self, params: Dict):
        null_action = np.zeros(3, )
        self._saved_data = self._scene.step(null_action, record_continuous_video=self._save_frames,
                                            img_size=self._save_cfg["img_size"])

    def execute_action(self, environment, state, action: Dict, **kwargs):
        target_pos = action["controlled_container_target_pos"]
        target_angle = action["controlled_container_target_angle"]
        curr_state = state
        for i in range(self._params["controller_max_horizon"]):
            curr_pos = curr_state["controlled_container_pos"]
            curr_angle = curr_state["controlled_container_angle"]
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
                random_angle = action_rng.uniform(low=self._params["action_range"]["angle"][0],
                                                  high=self._params["action_range"]["angle"][1])
                action = {"controlled_container_target_pos":   current_controlled_container_pos,
                          "controlled_container_target_angle": random_angle}
            else:
                random_x = action_rng.uniform(low=self._params["action_range"]["x"][0],
                                              high=self._params["action_range"]["x"][1])
                random_z = action_rng.uniform(low=self._params["action_range"]["z"][0],
                                              high=self._params["action_range"]["z"][1])
                action = {"controlled_container_target_pos":   np.array([random_x, random_z]),
                          "controlled_container_target_angle": 0}
            return action, False

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
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

    def randomize_environment(self, env_rng, params: Dict):
        self._scene.reset()

    @staticmethod
    def put_state_local_frame(state: Dict):
        return NotImplementedError

    @staticmethod
    def apply_local_action_at_state(state, local_action):
        raise NotImplementedError

    @staticmethod
    def add_action_noise(action: Dict, noise_rng: np.random.RandomState):
        raise NotImplementedError

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        raise NotImplementedError

    def get_state(self):
        state_vector = self._saved_data[0][0]
        # cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
        # self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
        # self._get_current_water_height(), in_poured_glass, in_control_glass])
        state = {"controlled_container_pos":   state_vector[:2],
                 "controlled_container_angle": state_vector[2],
                 "target_container_pos":       np.array([state_vector[6],0]),  # need to check this one
                 "control_volume":             state_vector[-1],
                 "target_volume":              state_vector[-2]}
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

    @staticmethod
    def put_state_local_frame_torch(state: Dict):
        raise NotImplementedError

    def simple_name(self):
        return "watering"

    def __repr__(self):
        return self.simple_name()

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        return True
