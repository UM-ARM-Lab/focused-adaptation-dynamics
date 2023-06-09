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

rope_key_name = 'rope'


class FloatingRopeScenario(ScenarioWithVisualization, MoveitPlanningSceneScenarioMixin):
    link_states_k = 'link_states'
    tinv_dim = 6  # SE(3)
    DISABLE_CDCPD = True
    IMAGE_H = 90
    IMAGE_W = 160
    n_links = 25
    KINECT_NAME = "kinect2"
    # FIXME: this is defined in multiple places
    state_keys = ['left_gripper', 'right_gripper', 'rope']
    action_keys = ['left_gripper_position', 'right_gripper_position']

    def __init__(self, params: Optional[dict] = None):
        ScenarioWithVisualization.__init__(self, params)
        MoveitPlanningSceneScenarioMixin.__init__(self, robot_namespace='')
        self.state_color_viz_pub = rospy.Publisher("state_color_viz", Image, queue_size=10, latch=True)
        self.state_depth_viz_pub = rospy.Publisher("state_depth_viz", Image, queue_size=10, latch=True)
        self.last_action = None
        self.gz = None
        self.get_rope_end_points_srv = rospy.ServiceProxy(ns_join(self.params['rope_name'], "get_dual_gripper_points"),
                                                          GetDualGripperPoints)
        self.get_rope_srv = rospy.ServiceProxy(ns_join(self.params['rope_name'], "get_rope_state"), GetRopeState,
                                               persistent=True)

        self.pos3d = Position3D()
        self.set_rope_state_srv = rospy.ServiceProxy(ns_join(self.params['rope_name'], "set_rope_state"), SetRopeState)
        self.reset_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

        self.get_links_states = Lazy(GetLinkStates)

        self.left_gripper_bbox_pub = rospy.Publisher('/left_gripper_bbox_pub', BoundingBox, queue_size=10, latch=True)
        self.right_gripper_bbox_pub = rospy.Publisher('/right_gripper_bbox_pub', BoundingBox, queue_size=10, latch=True)
        self.overstretching_srv = rospy.ServiceProxy(ns_join(self.params['rope_name'], "rope_overstretched"),
                                                     GetOverstretching)

        self.max_action_attempts = 100

        self.robot_reset_rng = np.random.RandomState(0)

    def needs_reset(self, state: Dict, params: Dict):
        return self.is_rope_overstretched()

    def is_rope_overstretched(self):
        try:
            res: GetOverstretchingResponse = self.overstretching_srv(GetOverstretchingRequest())
            return res.overstretched
        except Exception:
            return False

    def get_environment(self, params: Dict, **kwargs):
        extent = params['extent']
        res = params['res']
        shape = extent_to_env_shape(extent, res)
        origin_point = extent_res_to_origin_point(extent, res)
        env = {
            'res':          res,
            'origin_point': origin_point,
            'extent':       extent,
            'env':          np.zeros(shape, dtype=np.float32),
        }
        return env

    def hard_reset(self):
        self.reset_srv(EmptyRequest())

    def on_before_get_state_or_execute_action(self):
        self.on_before_action()

    def on_before_action(self):
        self.register_fake_grasping()

    def on_before_data_collection(self, params: Dict):
        self.on_before_action()

        self.gz = GazeboServices()

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
        self.execute_action(None, None, init_action, wait=True)

    def execute_action(self, environment, state, action: Dict, **kwargs):
        speed_mps = action.get('speed', 0.1)
        left_req = Position3DActionRequest(speed_mps=speed_mps,
                                           scoped_link_name=gz_scope(self.params['rope_name'], 'left_gripper'),
                                           position=ros_numpy.msgify(Point, action['left_gripper_position']))
        right_req = Position3DActionRequest(speed_mps=speed_mps,
                                            scoped_link_name=gz_scope(self.params['rope_name'], 'right_gripper'),
                                            position=ros_numpy.msgify(Point, action['right_gripper_position']))
        self.pos3d.set(left_req)
        self.pos3d.set(right_req)

        if kwargs.get("wait", True):
            wait_req = Position3DWaitRequest()
            wait_req.timeout_s = 10.0
            wait_req.scoped_link_names.append(gz_scope(self.params['rope_name'], 'left_gripper'))
            wait_req.scoped_link_names.append(gz_scope(self.params['rope_name'], 'right_gripper'))
            self.pos3d.wait(wait_req)

            rope_settling_time = action.get('settling_time', 1.0)
            rospy.sleep(rope_settling_time)

    def reset_rope(self, action_params: Dict):
        reset = SetRopeStateRequest()

        # TODO: rename this to rope endpoints reset positions or something
        reset.left_gripper.x = numpify(action_params['left_gripper_reset_position'][0])
        reset.left_gripper.y = numpify(action_params['left_gripper_reset_position'][1])
        reset.left_gripper.z = numpify(action_params['left_gripper_reset_position'][2])
        reset.right_gripper.x = numpify(action_params['right_gripper_reset_position'][0])
        reset.right_gripper.y = numpify(action_params['right_gripper_reset_position'][1])
        reset.right_gripper.z = numpify(action_params['right_gripper_reset_position'][2])

        self.set_rope_state_srv(reset)

    def dynamics_dataset_metadata(self):
        metadata = ScenarioWithVisualization.dynamics_dataset_metadata(self)
        kinect_pose = get_gazebo_kinect_pose()
        kinect_params = get_camera_params(self.KINECT_NAME)
        metadata.update({
            'world_to_rgb_optical_frame': self.tf.get_transform(parent='world', child='kinect2_rgb_optical_frame'),
            'kinect_pose':                ros_numpy.numpify(kinect_pose),
            'kinect_params':              kinect_params,
        })
        return metadata

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate: bool,
                      stateless: Optional[bool] = False):
        viz_action_sample_bbox(self.left_gripper_bbox_pub, get_action_sample_extent(action_params, 'left'))
        viz_action_sample_bbox(self.right_gripper_bbox_pub, get_action_sample_extent(action_params, 'right'))

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

            if not validate or self.is_action_valid(environment, state, action, action_params):
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

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        out_of_bounds = FloatingRopeScenario.grippers_out_of_bounds(action['left_gripper_position'],
                                                                    action['right_gripper_position'],
                                                                    action_params)

        max_gripper_d = default_if_none(action_params['max_distance_between_grippers'], 1000)
        gripper_d = np.linalg.norm(action['left_gripper_position'] - action['right_gripper_position'])
        too_far = gripper_d > max_gripper_d

        return not out_of_bounds and not too_far

    def local_planner_cost_function_torch(self, planner):
        def _local_planner_cost_function(actions: List[Dict],
                                         environment: Dict,
                                         goal_state: Dict,
                                         states: List[Dict]):
            del goal_state
            goal_cost = self.distance_to_goal(state=states[1], goal=planner.goal_region.goal, use_torch=True)
            action_cost = self.actions_cost_torch(states, actions, planner.action_params)
            return goal_cost * planner.params['goal_alpha'] + action_cost * planner.params['action_alpha']

        return _local_planner_cost_function

    def actions_cost(self, states: List[Dict], actions: List[Dict], action_params: Dict):
        cost = 0
        for state, action in zip(states, actions):
            max_gripper_delta = tf.cast(action_params['max_distance_gripper_can_move'], tf.float32)
            max_gripper_d = tf.cast(action_params['max_distance_between_grippers'], tf.float32)
            gripper_d = tf.linalg.norm(action['left_gripper_position'] - action['right_gripper_position'])
            left_gripper_delta = tf.linalg.norm(action['left_gripper_position'] - state['left_gripper'])
            right_gripper_delta = tf.linalg.norm(action['right_gripper_position'] - state['right_gripper'])
            left_gripper_delta_cost = tf.nn.relu(left_gripper_delta - max_gripper_delta)
            right_gripper_delta_cost = tf.nn.relu(right_gripper_delta - max_gripper_delta)
            max_gripper_d_cost = tf.nn.relu(gripper_d - max_gripper_d)
            cost_t = max_gripper_d_cost + left_gripper_delta_cost + right_gripper_delta_cost
            cost += cost_t

        return cost

    def actions_cost_torch(self, states: List[Dict], actions: List[Dict], action_params: Dict):
        import torch.nn.functional as F
        cost = 0
        for state, action in zip(states, actions):
            max_gripper_delta = action_params['max_distance_gripper_can_move']
            max_gripper_d = action_params['max_distance_between_grippers']
            gripper_d = (action['left_gripper_position'] - action['right_gripper_position']).norm()
            left_gripper_delta = (action['left_gripper_position'] - state['left_gripper']).norm()
            right_gripper_delta = (action['right_gripper_position'] - state['right_gripper']).norm()
            left_gripper_delta_cost = F.relu(left_gripper_delta - max_gripper_delta)
            right_gripper_delta_cost = F.relu(right_gripper_delta - max_gripper_delta)
            max_gripper_d_cost = F.relu(gripper_d - max_gripper_d)
            cost_t = max_gripper_d_cost + left_gripper_delta_cost + right_gripper_delta_cost
            cost += cost_t

        return cost

    @staticmethod
    def grippers_out_of_bounds(left_gripper, right_gripper, action_params: Dict):
        left_gripper_extent = action_params['left_gripper_action_sample_extent']
        right_gripper_extent = action_params['right_gripper_action_sample_extent']
        return is_out_of_bounds(left_gripper, left_gripper_extent) \
               or is_out_of_bounds(right_gripper, right_gripper_extent)

    @staticmethod
    def interpolate(start_state, end_state, step_size=0.05):
        left_gripper_start = np.array(start_state['left_gripper'])
        left_gripper_end = np.array(end_state['left_gripper'])

        right_gripper_start = np.array(start_state['right_gripper'])
        right_gripper_end = np.array(end_state['right_gripper'])

        left_gripper_delta = left_gripper_end - left_gripper_start
        right_gripper_delta = right_gripper_end - right_gripper_start

        left_gripper_steps = np.round(np.linalg.norm(left_gripper_delta) / step_size).astype(np.int64)
        right_gripper_steps = np.round(np.linalg.norm(right_gripper_delta) / step_size).astype(np.int64)
        steps = max(left_gripper_steps, right_gripper_steps, 1)

        interpolated_actions = []
        for t in np.linspace(step_size, 1, steps):
            left_gripper_t = left_gripper_start + left_gripper_delta * t
            right_gripper_t = right_gripper_start + right_gripper_delta * t
            action = {
                'left_gripper_position':  left_gripper_t,
                'right_gripper_position': right_gripper_t,
            }
            interpolated_actions.append(action)

        return interpolated_actions

    @staticmethod
    def robot_name():
        return "rope_3d"

    def randomize_environment(self, env_rng, params: Dict):
        pass

    @staticmethod
    def put_state_robot_frame(state: Dict):
        # Assumes everything is in robot frame already
        return {
            'left_gripper':  state['left_gripper'],
            'right_gripper': state['right_gripper'],
            rope_key_name:   state[rope_key_name],
        }

    @staticmethod
    def put_state_local_frame(state: Dict):
        rope = state[rope_key_name]
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        rope_points = tf.reshape(rope, rope_points_shape)

        center = tf.reduce_mean(rope_points, axis=-2)

        left_gripper_local = state['left_gripper'] - center
        right_gripper_local = state['right_gripper'] - center

        rope_points_local = rope_points - tf.expand_dims(center, axis=-2)
        rope_local = tf.reshape(rope_points_local, rope.shape)

        return {
            'left_gripper':  left_gripper_local,
            'right_gripper': right_gripper_local,
            rope_key_name:   rope_local,
        }

    @staticmethod
    def local_environment_center_differentiable(state):
        rope_vector = state[rope_key_name]
        rope_points = tf.reshape(rope_vector, [rope_vector.shape[0], -1, 3])
        center = tf.reduce_mean(rope_points, axis=1)
        return center

    @staticmethod
    def local_environment_center_differentiable_torch(state):
        rope_vector = state[rope_key_name]
        rope_points = rope_vector.reshape([rope_vector.shape[0], -1, 3])
        center = rope_points.mean(1)
        return center

    @staticmethod
    def apply_local_action_at_state(state, local_action):
        return {
            'left_gripper_position':  state['left_gripper'] + local_action['left_gripper_delta'],
            'right_gripper_position': state['right_gripper'] + local_action['right_gripper_delta']
        }

    @staticmethod
    def add_action_noise(action: Dict, noise_rng: np.random.RandomState):
        left_gripper_noise = noise_rng.normal(scale=0.01, size=[3])
        right_gripper_noise = noise_rng.normal(scale=0.01, size=[3])
        return {
            'left_gripper_position':  action['left_gripper_position'] + left_gripper_noise,
            'right_gripper_position': action['right_gripper_position'] + right_gripper_noise
        }

    @staticmethod
    def integrate_dynamics(s_t: Dict, delta_s_t: Dict):
        return {k: s_t[k] + delta_s_t[k] for k in s_t.keys()}

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_left_gripper_position = action['left_gripper_position']
        target_right_gripper_position = action['right_gripper_position']

        current_left_gripper_point = state['left_gripper']
        current_right_gripper_point = state['right_gripper']

        left_gripper_delta = target_left_gripper_position - current_left_gripper_point
        right_gripper_delta = target_right_gripper_position - current_right_gripper_point

        return {
            'left_gripper_delta':  left_gripper_delta,
            'right_gripper_delta': right_gripper_delta,
        }

    @staticmethod
    def state_to_gripper_position(state: Dict):
        gripper_position1 = np.reshape(state['left_gripper'], [3])
        gripper_position2 = np.reshape(state['right_gripper'], [3])
        return gripper_position1, gripper_position2

    def get_gazebo_rope_state(self):
        rope_res = self.get_rope_srv(GetRopeStateRequest())

        rope_state = []
        for p in rope_res.positions:
            rope_state.append(ros_numpy.numpify(p))
        rope_velocity = []
        for v in rope_res.velocities:
            rope_velocity.append(ros_numpy.numpify(v))
        rope_state = np.array(rope_state, np.float32)

        # transform into robot frame
        robot2world = self.tf.get_transform(self.root_link, 'world')
        rope_state_robot_frame = (robot2world @ homogeneous(rope_state).T).T[..., :-1]
        rope_state_robot_frame_vector = rope_state_robot_frame.flatten()

        return rope_state_robot_frame_vector

    def is_rope_point_attached(self, gripper: str):
        scoped_link_name = gz_scope(self.params['rope_name'], gripper + '_gripper')
        res: GetPosition3DResponse = self.pos3d.get(scoped_link_name=scoped_link_name)
        return res.enabled

    def get_rope_point_position(self, gripper: str):
        # NOTE: consider getting rid of this message type/service just use rope state [0] and rope state [-1]
        #  although that looses semantic meaning and means hard-coding indices a lot...
        scoped_link_name = gz_scope(self.params['rope_name'], gripper + '_gripper')
        res: GetPosition3DResponse = self.pos3d.get(scoped_link_name=scoped_link_name)
        rope_point_position = ros_numpy.numpify(res.pos)
        return rope_point_position

    def get_rope_point_positions(self):
        return self.get_rope_point_position('left'), self.get_rope_point_position('right')

    def get_state(self):
        gt_rope_vector = self.get_gazebo_rope_state()

        if self.DISABLE_CDCPD:
            cdcpd_state = {
                rope_key_name: np.array(gt_rope_vector, np.float32),
            }
        else:
            cdcpd_state = self.get_cdcpd_state.get_state()

        left_rope_point_position, right_rope_point_position = self.get_rope_point_positions()
        state = {
            'left_gripper':  np.array(left_rope_point_position, np.float32),
            'right_gripper': np.array(right_rope_point_position, np.float32),
            'gt_rope':       np.array(gt_rope_vector, np.float32),
        }
        state.update(cdcpd_state)
        state.update(self.get_links_states.get_state())
        return state

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        vec = state[rope_key_name]
        batch_shape = vec.shape[:-1]
        points = vec.reshape(batch_shape + (-1, 3))
        points_dense = densify_points(points.shape[0], points)
        return points_dense

    def __repr__(self):
        return "DualFloatingGripperRope"

    @staticmethod
    def simple_name():
        return "dual_floating"

    @staticmethod
    def distance_to_grippers_goal(state: Dict, goal: Dict, use_torch):
        left_gripper = state['left_gripper']
        right_gripper = state['right_gripper']
        distance1 = tf.linalg.norm(goal['left_gripper'] - left_gripper)
        distance2 = tf.linalg.norm(goal['right_gripper'] - right_gripper)
        return tf.math.maximum(distance1, distance2)

    @staticmethod
    def distance_grippers_and_any_point_goal(state: Dict, goal: Dict, use_torch):
        rope_points = tf.reshape(state[rope_key_name], [-1, 3])
        # well ok not _any_ node, but ones near the middle
        n_from_ends = 7
        distances = tf.linalg.norm(tf.expand_dims(goal['point'], axis=0) -
                                   rope_points, axis=1)[n_from_ends:-n_from_ends]
        rope_distance = tf.reduce_min(distances)

        left_gripper = tf.cast(state['left_gripper'], tf.float32)
        right_gripper = tf.cast(state['right_gripper'], tf.float32)
        distance_left = tf.linalg.norm(goal['left_gripper'] - left_gripper)
        distance_right = tf.linalg.norm(goal['right_gripper'] - right_gripper)
        d = tf.math.maximum(tf.math.maximum(distance_left, distance_right), rope_distance)

        return d

    @staticmethod
    def distance_grippers_and_any_point_goal2(state: Dict, goal: Dict, use_torch):
        n_from_ends = 7
        middle_rope_points = tf.reshape(state[rope_key_name], [-1, 3])[n_from_ends: -n_from_ends]
        middle_rope_point_distances = dist_to_bbox(point=middle_rope_points,
                                                   lower=goal['rope'][:3],
                                                   upper=goal['rope'][3:])
        rope_d = tf.reduce_min(middle_rope_point_distances)

        left_gripper_d = dist_to_bbox(point=state['left_gripper'],
                                      lower=goal['left_gripper'][:3],
                                      upper=goal['left_gripper'][3:])
        right_gripper_d = dist_to_bbox(point=state['right_gripper'],
                                       lower=goal['right_gripper'][:3],
                                       upper=goal['right_gripper'][3:])

        return tf.math.maximum(tf.math.maximum(left_gripper_d, right_gripper_d), rope_d)

    @staticmethod
    def distance_to_any_point_goal(state: Dict, goal: Dict, use_torch):
        if use_torch:
            import torch
            rope_points = state[rope_key_name].reshape([-1, 3])
            n_from_ends = 7
            distances = (torch.tensor(goal['point'])[None] - rope_points).norm(dim=1)[n_from_ends:-n_from_ends]
            min_distance = distances.min()
        else:
            rope_points = tf.reshape(state[rope_key_name], [-1, 3])
            # NOTE: well ok not _any_ node, but ones near the middle
            n_from_ends = 7
            distances = tf.linalg.norm(tf.expand_dims(goal['point'], axis=0) -
                                       rope_points, axis=1)[n_from_ends:-n_from_ends]
            min_distance = tf.reduce_min(distances)
        return min_distance

    @staticmethod
    def distance_to_midpoint_goal(state: Dict, goal: Dict, use_torch):
        rope_points = tf.reshape(state[rope_key_name], [-1, 3])
        rope_midpoint = rope_points[int(FloatingRopeScenario.n_links / 2)]
        distance = tf.linalg.norm(goal['midpoint'] - rope_midpoint)
        return distance

    def distance_to_goal(self, state: Dict, goal: Dict, use_torch=False):
        if goal['goal_type'] == 'midpoint':
            return self.distance_to_midpoint_goal(state, goal, use_torch)
        elif goal['goal_type'] == 'any_point':
            return self.distance_to_any_point_goal(state, goal, use_torch)
        elif goal['goal_type'] == 'grippers':
            return self.distance_to_grippers_goal(state, goal, use_torch)
        elif goal['goal_type'] == 'grippers_and_point':
            return self.distance_grippers_and_any_point_goal(state, goal, use_torch)
        elif goal['goal_type'] == 'grippers_and_point2':
            return self.distance_grippers_and_any_point_goal2(state, goal, use_torch)
        else:
            raise NotImplementedError()

    def goal_state_to_goal(self, goal_state: Dict, goal_type: str):
        if goal_type == 'midpoint':
            rope_points = tf.reshape(goal_state[rope_key_name], [-1, 3])
            rope_midpoint = rope_points[int(FloatingRopeScenario.n_links / 2)]
            return {
                'goal_type': goal_type,
                'midpoint':  rope_midpoint,
            }
        elif goal_type == 'any_point':
            # NOTE: since all points on the sampled ropes are the same point, it doesn't matter which one we pick here
            rope_points = tf.reshape(goal_state[rope_key_name], [-1, 3])
            rope_point = rope_points[0]
            return {
                'goal_type': goal_type,
                'point':     rope_point,
            }
        elif goal_type == 'grippers':
            left_gripper = goal_state['left_gripper']
            right_gripper = goal_state['right_gripper']
            return {
                'goal_type':     goal_type,
                'left_gripper':  left_gripper,
                'right_gripper': right_gripper,
            }
        elif goal_type == 'grippers_and_point':
            left_gripper = goal_state['left_gripper']
            right_gripper = goal_state['right_gripper']
            rope_points = tf.reshape(goal_state[rope_key_name], [-1, 3])
            rope_point = rope_points[0]
            return {
                'goal_type':     goal_type,
                'point':         rope_point,
                'left_gripper':  left_gripper,
                'right_gripper': right_gripper,
            }
        else:
            raise NotImplementedError()

    def classifier_distance_torch(self, s1: Dict, s2: Dict):
        import torch
        model_error = torch.norm(s1[rope_key_name] - s2[rope_key_name], dim=-1)
        return model_error

    def classifier_distance(self, s1: Dict, s2: Dict):
        model_error = np.linalg.norm(s1[rope_key_name] - s2[rope_key_name], axis=-1)
        # labeling_states = s1['rope']
        # labeling_predicted_states = s2['rope']
        # points_shape = labeling_states.shape.as_list()[:2] + [-1, 3]
        # labeling_points = tf.reshape(labeling_states, points_shape)
        # labeling_predicted_points = tf.reshape(labeling_predicted_states, points_shape)
        # model_error = tf.reduce_mean(tf.linalg.norm(labeling_points - labeling_predicted_points, axis=-1), axis=-1)
        return model_error

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

        if 'midpoint' in goal:
            midpoint_marker = Marker()
            midpoint_marker.scale.x = goal_threshold * 2
            midpoint_marker.scale.y = goal_threshold * 2
            midpoint_marker.scale.z = goal_threshold * 2
            midpoint_marker.action = Marker.ADD
            midpoint_marker.type = Marker.SPHERE
            midpoint_marker.header.frame_id = "robot_root"
            midpoint_marker.header.stamp = rospy.Time.now()
            midpoint_marker.ns = 'goal'
            midpoint_marker.id = 0
            midpoint_marker.color.r = r
            midpoint_marker.color.g = g
            midpoint_marker.color.b = b
            midpoint_marker.color.a = a
            midpoint_marker.pose.position.x = goal['midpoint'][0]
            midpoint_marker.pose.position.y = goal['midpoint'][1]
            midpoint_marker.pose.position.z = goal['midpoint'][2]
            midpoint_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(midpoint_marker)

        if 'point' in goal:
            point_marker = Marker()
            point_marker.scale.x = goal_threshold * 2
            point_marker.scale.y = goal_threshold * 2
            point_marker.scale.z = goal_threshold * 2
            point_marker.action = Marker.ADD
            point_marker.type = Marker.SPHERE
            point_marker.header.frame_id = "robot_root"
            point_marker.header.stamp = rospy.Time.now()
            point_marker.ns = 'goal'
            point_marker.id = 0
            point_marker.color.r = r
            point_marker.color.g = g
            point_marker.color.b = b
            point_marker.color.a = a
            point_marker.pose.position.x = goal['point'][0]
            point_marker.pose.position.y = goal['point'][1]
            point_marker.pose.position.z = goal['point'][2]
            point_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(point_marker)

        if 'left_gripper' in goal:
            left_gripper_marker = Marker()
            left_gripper_marker.scale.x = goal_threshold * 2
            left_gripper_marker.scale.y = goal_threshold * 2
            left_gripper_marker.scale.z = goal_threshold * 2
            left_gripper_marker.action = Marker.ADD
            left_gripper_marker.type = Marker.SPHERE
            left_gripper_marker.header.frame_id = "robot_root"
            left_gripper_marker.header.stamp = rospy.Time.now()
            left_gripper_marker.ns = 'goal'
            left_gripper_marker.id = 1
            left_gripper_marker.color.r = r
            left_gripper_marker.color.g = g
            left_gripper_marker.color.b = b
            left_gripper_marker.color.a = a
            left_gripper_marker.pose.position.x = goal['left_gripper'][0]
            left_gripper_marker.pose.position.y = goal['left_gripper'][1]
            left_gripper_marker.pose.position.z = goal['left_gripper'][2]
            left_gripper_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(left_gripper_marker)

        if 'right_gripper' in goal:
            right_gripper_marker = Marker()
            right_gripper_marker.scale.x = goal_threshold * 2
            right_gripper_marker.scale.y = goal_threshold * 2
            right_gripper_marker.scale.z = goal_threshold * 2
            right_gripper_marker.action = Marker.ADD
            right_gripper_marker.type = Marker.SPHERE
            right_gripper_marker.header.frame_id = "robot_root"
            right_gripper_marker.header.stamp = rospy.Time.now()
            right_gripper_marker.ns = 'goal'
            right_gripper_marker.id = 2
            right_gripper_marker.color.r = r
            right_gripper_marker.color.g = g
            right_gripper_marker.color.b = b
            right_gripper_marker.color.a = a
            right_gripper_marker.pose.position.x = goal['right_gripper'][0]
            right_gripper_marker.pose.position.y = goal['right_gripper'][1]
            right_gripper_marker.pose.position.z = goal['right_gripper'][2]
            right_gripper_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(right_gripper_marker)

        self.state_viz_pub.publish(goal_marker_msg)

    def plot_goal_boxes(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
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

        if 'point_box' in goal:
            point_marker = make_box_marker_from_extents(goal['point_box'])
            point_marker.header.frame_id = "robot_root"
            point_marker.header.stamp = rospy.Time.now()
            point_marker.ns = 'goal'
            point_marker.id = 0
            point_marker.color.r = r
            point_marker.color.g = g
            point_marker.color.b = b
            point_marker.color.a = a
            goal_marker_msg.markers.append(point_marker)

        if 'left_gripper_box' in goal:
            left_gripper_marker = make_box_marker_from_extents(goal['left_gripper_box'])
            left_gripper_marker.header.frame_id = "robot_root"
            left_gripper_marker.header.stamp = rospy.Time.now()
            left_gripper_marker.ns = 'goal'
            left_gripper_marker.id = 1
            left_gripper_marker.color.r = r
            left_gripper_marker.color.g = g
            left_gripper_marker.color.b = b
            left_gripper_marker.color.a = a
            goal_marker_msg.markers.append(left_gripper_marker)

        if 'right_gripper_box' in goal:
            right_gripper_marker = make_box_marker_from_extents(goal['right_gripper_box'])
            right_gripper_marker.header.frame_id = "robot_root"
            right_gripper_marker.header.stamp = rospy.Time.now()
            right_gripper_marker.ns = 'goal'
            right_gripper_marker.id = 2
            right_gripper_marker.color.r = r
            right_gripper_marker.color.g = g
            right_gripper_marker.color.b = b
            right_gripper_marker.color.a = a
            goal_marker_msg.markers.append(right_gripper_marker)

        self.state_viz_pub.publish(goal_marker_msg)

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        return dynamics_loss_function(dataset_element, predictions)

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        return dynamics_points_metrics_function(dataset_element, predictions)

    def plot_environment_rviz(self, environment: Dict, **kwargs):
        ScenarioWithVisualization.plot_environment_rviz(self, environment, **kwargs)
        MoveitPlanningSceneScenarioMixin.plot_environment_rviz(self, environment, **kwargs)

    def plot_state_rviz(self, state: Dict, **kwargs):
        label = kwargs.get("label", "")
        color_msg = ColorRGBA(*colors.to_rgba(kwargs.get("color", "r")))
        s = kwargs.get('s', 0.007)
        if 'a' in kwargs:
            color_msg.a = kwargs['a']
            a = kwargs['a']
        else:
            a = 1.0
        idx = kwargs.get("idx", 0)

        msg = MarkerArray()

        ig = marker_index_generator(idx)

        if 'gt_rope' in state:
            rope_points = np.reshape(state['gt_rope'], [-1, 3])
            markers = make_rope_marker(rope_points, 'robot_root', label + "_gt_rope", next(ig), color_msg, s)
            msg.markers.extend(markers)

        if 'rope' in state:
            rope_points = np.reshape(state['rope'], [-1, 3])
            markers = make_rope_marker(rope_points, 'robot_root', label + "_rope", next(ig),
                                       adjust_lightness_msg(color_msg, 0.9), s)
            msg.markers.extend(markers)

        if add_predicted('rope') in state:
            rope_points = np.reshape(state[add_predicted('rope')], [-1, 3])
            markers = make_rope_marker(rope_points, 'robot_root', label + "_pred_rope", next(ig), color_msg, s,
                                       Marker.CUBE_LIST)
            msg.markers.extend(markers)

        if 'left_gripper' in state:
            left_gripper_sphere = make_gripper_marker(state['left_gripper'], next(ig), ColorRGBA(0.2, 0.2, 0.8, a),
                                                      label + '_l', Marker.SPHERE)
            msg.markers.append(left_gripper_sphere)

        if 'right_gripper' in state:
            right_gripper_sphere = make_gripper_marker(state['right_gripper'], next(ig), ColorRGBA(0.8, 0.8, 0.2, a),
                                                       label + "_r", Marker.SPHERE)
            msg.markers.append(right_gripper_sphere)

        if add_predicted('left_gripper') in state:
            lgpp = state[add_predicted('left_gripper')]
            left_gripper_sphere = make_gripper_marker(lgpp, next(ig), ColorRGBA(0.2, 0.2, 0.8, a), label + "_lp",
                                                      Marker.CUBE)
            msg.markers.append(left_gripper_sphere)

        if add_predicted('right_gripper') in state:
            rgpp = state[add_predicted('right_gripper')]
            right_gripper_sphere = make_gripper_marker(rgpp, next(ig), ColorRGBA(0.8, 0.8, 0.2, a), label + "_rp",
                                                       Marker.CUBE)
            msg.markers.append(right_gripper_sphere)

        s = kwargs.get("scale", 1.0)
        msg = scale_marker_array(msg, s)

        self.state_viz_pub.publish(msg)

        if in_maybe_predicted('rgbd', state):
            publish_color_image(self.state_color_viz_pub, state['rgbd'][:, :, :3])
            publish_depth_image(self.state_depth_viz_pub, state['rgbd'][:, :, 3])

        if add_predicted('stdev') in state:
            stdev_t = state[add_predicted('stdev')][0]
            self.plot_stdev(stdev_t)

        if 'error' in state:
            error_t = state['error']
            self.plot_error_rviz(error_t)

    def plot_action_rviz(self, state: Dict, action: Dict, label: str = 'action', **kwargs):
        state_action = {}
        state_action.update(state)
        state_action.update(action)
        self.plot_action_rviz_internal(state_action, label=label, **kwargs)

    def plot_action_rviz_internal(self, data: Dict, label: str, **kwargs):
        s1 = np.reshape(get_maybe_predicted(data, 'left_gripper'), [3])
        s2 = np.reshape(get_maybe_predicted(data, 'right_gripper'), [3])
        a1 = np.reshape(get_maybe_predicted(data, 'left_gripper_position'), [3])
        a2 = np.reshape(get_maybe_predicted(data, 'right_gripper_position'), [3])

        idx = kwargs.pop("idx", None)
        ig = marker_index_generator(idx)
        if idx is not None:
            idx1 = next(ig)
            idx2 = next(ig)
        else:
            idx1 = kwargs.pop("idx1", 0)
            idx2 = kwargs.pop("idx2", 1)

        scale = kwargs.pop("scale", 2.0)

        msg = MarkerArray()
        msg.markers.append(rviz_arrow(s1, a1, idx=idx1, label=label, **kwargs, scale=scale, frame_id='robot_root'))
        msg.markers.append(rviz_arrow(s2, a2, idx=idx2, label=label, **kwargs, scale=scale, frame_id='robot_root'))

        self.action_viz_pub.publish(msg)

    def register_fake_grasping(self):
        register_left_req = RegisterPosition3DControllerRequest()
        register_left_req.scoped_link_name = gz_scope(self.params['rope_name'], "left_gripper")
        register_left_req.controller_type = "kinematic"
        register_left_req.position_only = True
        register_left_req.fixed_rot = True
        self.pos3d.register(register_left_req)
        register_right_req = RegisterPosition3DControllerRequest()
        register_right_req.scoped_link_name = gz_scope(self.params['rope_name'], "right_gripper")
        register_right_req.controller_type = "kinematic"
        register_right_req.position_only = True
        register_right_req.fixed_rot = True
        self.pos3d.register(register_right_req)

    def make_rope_endpoint_follow_gripper(self, side: str):
        follow_req = Position3DFollowRequest()
        follow_req.scoped_link_name = gz_scope(self.params['rope_name'], f"{side}_gripper")
        follow_req.frame_id = f"{side}_tool"
        self.pos3d.follow(follow_req)

    def make_rope_endpoints_follow_gripper(self):
        self.make_rope_endpoint_follow_gripper('left')
        self.make_rope_endpoint_follow_gripper('right')

    def make_simple_grippers_marker(self, example: Dict, id: int):
        msg = Marker()
        msg.header.frame_id = 'robot_root'
        msg.type = Marker.SPHERE_LIST
        msg.action = Marker.ADD
        msg.id = id
        msg.color.g = 1
        msg.color.a = 1
        msg.scale.x = 0.01
        msg.scale.y = 0.01
        msg.scale.z = 0.01
        left_gripper_vec3 = ros_numpy.msgify(Vector3, example['left_gripper'][0])
        right_gripper_vec3 = ros_numpy.msgify(Vector3, example['right_gripper'][0])
        msg.points.append(left_gripper_vec3)
        msg.points.append(right_gripper_vec3)
        return msg

    def debug_viz_state_action(self, inputs, b, label: str, color='red', use_predicted: bool = True):
        state_0 = numpify({k: inputs[add_predicted_cond(k, use_predicted)][b, 0] for k in self.state_keys})
        action_0 = numpify({k: inputs[k][b, 0] for k in self.action_keys})
        state_1 = numpify({k: inputs[add_predicted_cond(k, use_predicted)][b, 1] for k in self.state_keys})
        self.plot_state_rviz(state_0, idx=0, label=label, color=color)
        self.plot_state_rviz(state_1, idx=1, label=label, color=color)
        self.plot_action_rviz(state_0, action_0, idx=1, label=label, color=color)
        if 'is_close' in inputs:
            self.plot_is_close(inputs['is_close'][b, 1])
        if 'error' in inputs:
            error_t = inputs['error'][b, 1]
            self.plot_error_rviz(error_t)

    def transformation_params_to_matrices(self, obj_transforms):
        return xyzrpy_to_matrices(obj_transforms)

    def compute_obj_points(self, inputs: Dict, num_object_interp: int, batch_size: int, use_predicted: bool = True):
        keys = ['rope', 'left_gripper', 'right_gripper']
        if use_predicted:
            keys = [add_predicted(k) for k in keys]

        def _make_points(k, t):
            v = inputs[k][:, t]
            points = tf.reshape(v, [batch_size, -1, 3])
            points = densify_points(batch_size, points)
            return points

        obj_points_0 = {k: _make_points(k, 0) for k in keys}
        obj_points_1 = {k: _make_points(k, 1) for k in keys}

        def _linspace(k):
            return tf.linspace(obj_points_0[k], obj_points_1[k], num_object_interp, axis=1)

        swept_obj_points = tf.concat([_linspace(k) for k in keys], axis=2)

        # TODO: include the robot as an object here?
        # obj_points = tf.concat([robot_points, swept_obj_points], axis=1)
        obj_points = tf.expand_dims(swept_obj_points, axis=1)

        return obj_points

    def aug_apply_no_ik(self,
                        moved_mask,
                        m,
                        to_local_frame,
                        inputs: Dict,
                        batch_size,
                        time,
                        h: int,
                        w: int,
                        c: int,
                        use_predicted: bool = True,
                        visualize: bool = False,
                        *args,
                        **kwargs,
                        ):
        """

        Args:
            m: [b, k, 4, 4]
            to_local_frame: [b, 3]  the 1 can also be equal to time
            inputs:
            batch_size:
            time:
            h:
            w:
            c:

        Returns:

        """
        to_local_frame_expanded = to_local_frame[:, None, None]
        m_expanded = m[:, None]

        # apply those to the rope and grippers
        rope_k = add_predicted_cond('rope', use_predicted)
        rope_points = tf.reshape(inputs[rope_k], [batch_size, time, -1, 3])
        left_gripper_k = add_predicted_cond('left_gripper', use_predicted)
        left_gripper_point = inputs[left_gripper_k]
        right_gripper_k = add_predicted_cond('right_gripper', use_predicted)
        right_gripper_point = inputs[right_gripper_k]
        left_gripper_points = tf.expand_dims(left_gripper_point, axis=-2)
        right_gripper_points = tf.expand_dims(right_gripper_point, axis=-2)

        def _transform(_m, points, _to_local_frame):
            points_local_frame = points - _to_local_frame
            points_local_frame_aug = transform_points_3d(_m, points_local_frame)
            return points_local_frame_aug + _to_local_frame

        # m is expanded to broadcast across batch & num_points dimensions
        rope_points_aug = _transform(m_expanded, rope_points, to_local_frame_expanded)
        left_gripper_points_aug = _transform(m_expanded, left_gripper_points, to_local_frame_expanded)
        right_gripper_points_aug = _transform(m_expanded, right_gripper_points, to_local_frame_expanded)

        # compute the new action
        left_gripper_position = inputs['left_gripper_position']
        right_gripper_position = inputs['right_gripper_position']
        # m is expanded to broadcast across batch dimensions
        left_gripper_position_aug = _transform(m, left_gripper_position, tf.expand_dims(to_local_frame, -2))
        right_gripper_position_aug = _transform(m, right_gripper_position, tf.expand_dims(to_local_frame, -2))

        rope_aug = tf.reshape(rope_points_aug, [batch_size, time, -1])
        left_gripper_aug = tf.reshape(left_gripper_points_aug, [batch_size, time, -1])
        right_gripper_aug = tf.reshape(right_gripper_points_aug, [batch_size, time, -1])

        # Now that we've updated the state/action in inputs, compute the local origin point
        state_aug_0 = {
            'left_gripper':  left_gripper_aug[:, 0],
            'right_gripper': right_gripper_aug[:, 0],
            'rope':          rope_aug[:, 0]
        }
        local_center_aug = self.local_environment_center_differentiable(state_aug_0)
        res = tf.cast(inputs['res'], tf.float32)
        local_origin_point_aug = batch_center_res_shape_to_origin_point(local_center_aug, res, h, w, c)

        object_aug_update = {
            rope_k:                   rope_aug,
            left_gripper_k:           left_gripper_aug,
            right_gripper_k:          right_gripper_aug,
            'left_gripper_position':  left_gripper_position_aug,
            'right_gripper_position': right_gripper_position_aug,
        }

        # also apply m to the link states, if they are present
        has_link_states = self.link_states_k in inputs
        if has_link_states:
            link_states = inputs[self.link_states_k]
            link_states_aug = []
            for b in range(batch_size):
                link_states_aug_b = []
                link_states_b = link_states[b]
                for t in range(time):
                    link_states_b_t = link_states_b[t]
                    link_states_b_t_aug = transform_link_states(m[b], link_states_b_t)
                    link_states_aug_b.append(link_states_b_t_aug)
                link_states_aug.append(link_states_aug_b)
            link_states_aug = np.array(link_states_aug)
            object_aug_update[self.link_states_k] = link_states_aug

        if visualize:
            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':          inputs['env'][b],
                    'res':          res[b],
                    'extent':       inputs['extent'][b],
                    'origin_point': inputs['origin_point'][b],
                }

                self.plot_environment_rviz(env_b)
                state_0 = numpify({k: inputs[add_predicted_cond(k, use_predicted)][b, 0] for k in self.state_keys})
                action_0 = numpify({k: inputs[k][b, 0] for k in self.action_keys})
                self.plot_state_rviz(state_0, idx=0, label='apply_aug', color='pink')
                self.plot_action_rviz(state_0, action_0, idx=1, label='apply_aug', color='white')
        return object_aug_update, local_origin_point_aug, local_center_aug

    @staticmethod
    def tinv_sample_transform(rng, scaling, a=0.25):
        lower = np.array([-a, -a, -a, -np.pi, -np.pi, -np.pi])
        upper = np.array([a, a, a, np.pi, np.pi, np.pi])
        transform = rng.uniform(lower, upper).astype(np.float32) * scaling
        return transform

    def tinv_set_state(self, params, state_rng, visualize):
        self.randomize_environment(state_rng, params)
        # this just basically sets a random-ish state by taking random actions
        _params = deepcopy(params)
        _params['steps_per_traj'] = 10
        collect_trajectory(params=_params,
                           scenario=self,
                           traj_idx=0,
                           predetermined_start_state=None,
                           predetermined_actions=None,
                           verbose=(1 if visualize else 0),
                           action_rng=state_rng)

    def tinv_generate_data(self, action_rng: np.random.RandomState, params, visualize):
        example, invalid = collect_trajectory(params=params,
                                              scenario=self,
                                              traj_idx=0,
                                              predetermined_start_state=None,
                                              predetermined_actions=None,
                                              verbose=(1 if visualize else 0),
                                              action_rng=action_rng)
        return example, invalid

    def tinv_generate_data_from_aug_pred(self, params, example_aug_pred, visualize):
        unused_rng = np.random.RandomState(0)
        action_ks = ['left_gripper_position', 'right_gripper_position']
        predetermined_action = {}
        for k in action_ks:
            predetermined_action[k] = example_aug_pred[k][0]  # is there a time dim? or batch?

        example_aug_actual, invalid = collect_trajectory(params=params,
                                                         scenario=self,
                                                         traj_idx=0,
                                                         predetermined_start_state=None,
                                                         predetermined_actions=[predetermined_action],
                                                         verbose=(1 if visualize else 0),
                                                         action_rng=unused_rng)
        return example_aug_actual, invalid

    def tinv_apply_transformation(self, example: Dict, transform, visualize):
        time = 2

        example = coerce_types(example)
        example_aug = deepcopy(example)

        example_batch = add_batch(example)  # add batch
        obj_points = self.compute_obj_points(example_batch, num_object_interp=1, batch_size=1, use_predicted=False)
        moved_mask = compute_moved_mask(obj_points)

        m = self.transformation_params_to_matrices(tf.convert_to_tensor(add_batch(transform), tf.float32))
        to_local_frame = get_local_frame(moved_mask, obj_points)
        example_aug_update, _, _ = self.aug_apply_no_ik(moved_mask, m, to_local_frame, example_batch,
                                                        batch_size=1, time=time, visualize=visualize,
                                                        use_predicted=False,
                                                        h=64,  # FIXME: hardcoded!!!
                                                        w=64,
                                                        c=64,
                                                        )
        example_aug_update = remove_batch(example_aug_update)

        example_aug = nested_dict_update(example_aug, example_aug_update)
        return example_aug, moved_mask

    def tinv_set_state_from_aug_pred(self, params, example_aug_pred, moved_mask, visualize):
        # set the simulator state to make the augmented state
        link_states_aug_pred_w_time = example_aug_pred[self.link_states_k]
        link_states_aug_pred = link_states_aug_pred_w_time[0]

        restore_gazebo(self.gz, link_states_aug_pred, self)

        if visualize:
            state = self.get_state()
            self.plot_state_rviz(state, label='tinv_set_state', color='m')

        return (invalid := False)

    def tinv_error(self, example: Dict, example_aug: Dict, moved_mask):
        error = self.classifier_distance(example, example_aug)[-1]
        return error

    @staticmethod
    def put_state_local_frame_torch(state: Dict):
        rope = state[rope_key_name]
        rope_points_shape = rope.shape[:-1] + (-1, 3)
        rope_points = rope.reshape(rope_points_shape)

        center = rope_points.mean(-2)

        left_gripper_local = state['left_gripper'] - center
        right_gripper_local = state['right_gripper'] - center

        rope_points_local = rope_points - center.unsqueeze(-2)
        rope_local = rope_points_local.reshape(rope.shape)

        return {
            'left_gripper':  left_gripper_local,
            'right_gripper': right_gripper_local,
            rope_key_name:   rope_local,
        }

    def example_dict_to_flat_vector(self, example):
        import torch
        rope = example[add_predicted('rope')]
        left_gripper = example[add_predicted('left_gripper')]
        right_gripper = example[add_predicted('right_gripper')]
        batch_size = rope.shape[0]
        transition = torch.cat([rope, left_gripper, right_gripper], dim=-1)  # [batch_size, 2, 81]
        return transition.reshape([batch_size, -1])

    def flat_vector_to_example_dict(self, example, flat_vector_aug):
        batch_size = example['env'].shape[0]
        transition = flat_vector_aug.reshape([batch_size, 2, 81])
        rope = transition[..., :75]
        left_gripper = transition[..., 75:75 + 3]
        right_gripper = transition[..., 75 + 3:75 + 6]
        example_aug = deepcopy(example)
        example_aug[add_predicted('rope')] = rope
        example_aug[add_predicted('left_gripper')] = left_gripper
        example_aug[add_predicted('right_gripper')] = right_gripper
        return example_aug
