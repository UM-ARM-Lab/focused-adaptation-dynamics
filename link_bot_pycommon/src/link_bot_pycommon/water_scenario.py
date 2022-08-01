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

from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization

control_box_name = "control_box"

class WaterSimScenario(ScenarioWithVisualization):
    """
    Representation:
    state: target/control pose, target/control volume
    goal: target_volume, tolerance
    action: goal x,z of control. theta, which can be 0.
    """
    def __init__(self, params: Optional[dict] = None):
        ScenarioWithVisualization.__init__(self, params)
        self.state_color_viz_pub = rospy.Publisher("state_color_viz", Image, queue_size=10, latch=True)
        self.state_depth_viz_pub = rospy.Publisher("state_depth_viz", Image, queue_size=10, latch=True)
        self.last_action = None
        self.pos3d = Position3D()
        self.max_action_attempts = 100
        self.robot_reset_rng = np.random.RandomState(0)
        self._make_softgym_env()

    def _make_softgym_env(self):
        pass

    def needs_reset(self, state: Dict, params: Dict):
        raise NotImplementedError()


    def get_environment(self, params: Dict, **kwargs):
        env = {"env":np.zeros((40,40,40))}
        return env

    def hard_reset(self):
        raise NotImplementedError

    def on_before_get_state_or_execute_action(self):
        self.on_before_action()

    def on_before_action(self):
        self.register_fake_grasping()

    def on_before_data_collection(self, params: Dict):
        self.on_before_action()
        init_action = {}
        self.execute_action(None, None, init_action, wait=True)

    def execute_action(self, environment, state, action: Dict, **kwargs):
        pass


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
        current_controlled_container_pos = state["controller_container_pose"]
        for _ in range(self.max_action_attempts):
            is_pour = action_rng.randint(2)
            if is_pour:
                random_angle = action_rng.random()
                action = {"controlled_container_target_pos":current_controlled_container_pos,
                          "controlled_container_angle": random_angle}
            else:
                random_delta = action_rng.uniform(low=0.01, high=0.01, size=(2,))
                action = {"controlled_container_target_pos": current_controlled_container_pos + random_delta,
                          "controlled_container_angle": 0}
            return action


    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        return True

    def actions_cost(self, states: List[Dict], actions: List[Dict], action_params: Dict):
        return 0

    def actions_cost_torch(self, states: List[Dict], actions: List[Dict], action_params: Dict):
        return 0

    @staticmethod
    def robot_name():
        return "control_box"

    def randomize_environment(self, env_rng, params: Dict):
        pass

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
        state = {"controlled_container_pos":np.array([0,1]),
                 "target_container_pos": np.array([0,1]),
                 "control_volume":0.01,
                 "target_volume":0.99}
        return state


    def distance_to_goal(self, state: Dict, goal: Dict, use_torch=False):
        goal_target_volume = goal["goal_target_volume"]
        return abs(state["target_volume"]-goal_target_volume)

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

