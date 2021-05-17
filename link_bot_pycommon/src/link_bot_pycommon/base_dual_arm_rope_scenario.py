import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import rosnode
from arm_robots.robot_utils import merge_joint_state_and_scene_msg
from link_bot_data.dataset_utils import add_predicted, deserialize_scene_msg, _deserialize_scene_msg
from link_bot_pycommon.dual_arm_get_gripper_positions import DualArmGetGripperPositions
from link_bot_pycommon.grid_utils import batch_point_to_idx_tf_3d_in_batched_envs, batch_idx_to_point_3d_in_env_tf
from link_bot_pycommon.moveit_planning_scene_mixin import MoveitPlanningSceneScenarioMixin
from link_bot_pycommon.moveit_utils import make_joint_state
from merrrt_visualization.rviz_animation_controller import RvizSimpleStepper
from moonshine.geometry import rotate_points_3d, make_rotation_matrix_like
from moonshine.moonshine_utils import numpify
from moveit_msgs.msg import RobotState, RobotTrajectory, PlanningScene
from std_msgs.msg import Float32
from trajectory_msgs.msg import JointTrajectoryPoint

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import moveit_commander
    from arm_robots.get_robot import get_moveit_robot

import rospy
from arc_utilities.listener import Listener
from geometry_msgs.msg import PoseStamped, Point
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from arm_gazebo_msgs.srv import ExcludeModels, ExcludeModelsRequest, ExcludeModelsResponse
from rosgraph.names import ns_join
from sensor_msgs.msg import JointState, PointCloud2
from tf.transformations import quaternion_from_euler

DEBUG_VIZ_STATE_AUG = True
SHOW_ALL = False


def debug_viz_batch_indices(batch_size):
    if SHOW_ALL:
        return range(batch_size)
    else:
        return [0]


def get_joint_positions_given_state_and_plan(plan: RobotTrajectory, robot_state: RobotState):
    if len(plan.joint_trajectory.points) == 0:
        predicted_joint_positions = robot_state.joint_state.position
    else:
        final_point: JointTrajectoryPoint = plan.joint_trajectory.points[-1]
        predicted_joint_positions = []
        for joint_name in robot_state.joint_state.name:
            if joint_name in plan.joint_trajectory.joint_names:
                joint_idx_in_final_point = plan.joint_trajectory.joint_names.index(joint_name)
                joint_position = final_point.positions[joint_idx_in_final_point]
            elif joint_name in robot_state.joint_state.name:
                joint_idx_in_state = list(robot_state.joint_state.name).index(joint_name)
                joint_position = float(robot_state.joint_state.position[joint_idx_in_state])
            else:
                raise ValueError(f"joint {joint_name} is in neither the start state nor the the planed trajectory")
            predicted_joint_positions.append(joint_position)
    return predicted_joint_positions


def robot_state_msg_from_state_dict(state: Dict):
    robot_state = RobotState()
    robot_state.joint_state = joint_state_msg_from_state_dict(state)
    robot_state.joint_state.velocity = [0.0] * len(robot_state.joint_state.name)
    aco = state.get('attached_collision_objects', None)
    if aco is not None:
        robot_state.attached_collision_objects = aco
    return robot_state


def joint_state_msg_from_state_dict(state: Dict):
    joint_state = JointState(position=state['joint_positions'], name=to_list_of_strings(state['joint_names']))
    joint_state.header.stamp = rospy.Time.now()
    return joint_state


def joint_state_msg_from_state_dict_predicted(state: Dict):
    joint_state = JointState(position=state[add_predicted('joint_positions')],
                             name=to_list_of_strings(state['joint_names']))
    joint_state.header.stamp = rospy.Time.now()
    return joint_state


def to_list_of_strings(x):
    if isinstance(x[0], bytes):
        return [n.decode("utf-8") for n in x]
    elif isinstance(x[0], str):
        return [str(n) for n in x]
    elif isinstance(x, tf.Tensor):
        return [n.decode("utf-8") for n in x.numpy()]
    else:
        raise NotImplementedError()


def to_point_msg(v):
    return Point(x=v[0], y=v[1], z=v[2])


class BaseDualArmRopeScenario(FloatingRopeScenario, MoveitPlanningSceneScenarioMixin):
    DISABLE_CDCPD = True
    ROPE_NAMESPACE = 'rope_3d'

    def __init__(self, robot_namespace: str):
        FloatingRopeScenario.__init__(self)
        MoveitPlanningSceneScenarioMixin.__init__(self, robot_namespace)

        self.robot_namespace = robot_namespace
        self.service_provider = BaseServices()
        joint_state_viz_topic = ns_join(self.robot_namespace, "joint_states_viz")
        self.joint_state_viz_pub = rospy.Publisher(joint_state_viz_topic, JointState, queue_size=10)
        self.cdcpd_listener = Listener("cdcpd/output", PointCloud2)

        # NOTE: you may want to override this for your specific robot/scenario
        self.left_preferred_tool_orientation = quaternion_from_euler(np.pi, 0, 0)
        self.right_preferred_tool_orientation = quaternion_from_euler(np.pi, 0, 0)

        self.size_of_box_around_tool_for_planning = 0.05
        exclude_srv_name = ns_join(self.robot_namespace, "exclude_models_from_planning_scene")
        self.exclude_from_planning_scene_srv = rospy.ServiceProxy(exclude_srv_name, ExcludeModels)
        # FIXME: this blocks until the robot is available, we need lazy construction
        self.robot = get_moveit_robot(self.robot_namespace, raise_on_failure=True)

        self.get_gripper_positions = DualArmGetGripperPositions(self.robot)

    def add_boxes_around_tools(self):
        # add attached collision object to prevent moveit from smooshing the rope and ends of grippers into obstacles
        self.moveit_scene = moveit_commander.PlanningSceneInterface(ns=self.robot_namespace)
        self.robust_add_to_scene(self.robot.left_tool_name, 'left_tool_box',
                                 self.robot.get_left_gripper_links() + ['end_effector_left', 'left_tool'])
        self.robust_add_to_scene(self.robot.right_tool_name, 'right_tool_box',
                                 self.robot.get_right_gripper_links() + ['end_effector_right', 'right_tool'])

    def robust_add_to_scene(self, link: str, new_object_name: str, touch_links: List[str]):
        box_pose = PoseStamped()
        box_pose.header.frame_id = link
        box_pose.pose.orientation.w = 1.0
        box_size = self.size_of_box_around_tool_for_planning
        while True:
            # self.moveit_scene.add_box(new_object_name, box_pose, size=(box_size, box_size, box_size))
            self.moveit_scene.add_sphere(new_object_name, box_pose, radius=box_size)
            self.moveit_scene.attach_box(link, new_object_name, touch_links=touch_links)

            rospy.sleep(0.1)

            # Test if the box is in attached objects
            attached_objects = self.moveit_scene.get_attached_objects([new_object_name])
            is_attached = len(attached_objects.keys()) > 0

            # Note that attaching the box will remove it from known_objects
            is_known = new_object_name in self.moveit_scene.get_known_object_names()

            if is_attached and not is_known:
                break

    def on_before_get_state_or_execute_action(self):
        self.robot.connect()

        self.add_boxes_around_tools()

        # Mark the rope as a not-obstacle
        exclude = ExcludeModelsRequest()
        exclude.model_names.append("rope_3d")
        exclude.model_names.append(self.robot_namespace)
        self.exclude_from_planning_scene_srv(exclude)

    def on_before_data_collection(self, params: Dict):
        self.on_before_get_state_or_execute_action()
        self.add_boxes_around_tools()

        # Set the preferred tool orientations
        self.robot.store_tool_orientations({
            self.robot.left_tool_name:  self.left_preferred_tool_orientation,
            self.robot.right_tool_name: self.right_preferred_tool_orientation,
        })

    def get_n_joints(self):
        return len(self.get_joint_names())

    def get_joint_names(self):
        return self.robot.get_joint_names()

    def get_state(self):
        # TODO: this should be composed of function calls to get_state for arm_no_rope and get_state for rope?
        joint_state: JointState = self.robot._joint_state_listener.get()

        # FIXME: "Joint values for monitored state are requested but the full state is not known"
        # for _ in range(5):
        #     left_gripper_position, right_gripper_position = self.robot.get_gripper_positions()
        #     rospy.sleep(0.02)

        # rgbd = self.get_rgbd()

        gt_rope_state_vector = self.get_rope_state()
        gt_rope_state_vector = np.array(gt_rope_state_vector, np.float32)

        if self.DISABLE_CDCPD:
            cdcpd_rope_state_vector = gt_rope_state_vector
        else:
            cdcpd_rope_state_vector = self.get_cdcpd_state()

        state = {
            'joint_positions': np.array(joint_state.position, np.float32),
            'joint_names':     np.array(joint_state.name),
            # 'rgbd':            rgbd,
            'gt_rope':         gt_rope_state_vector,
            'rope':            cdcpd_rope_state_vector,
        }
        state.update(self.get_gripper_positions.get_state())
        return state

    def states_description(self) -> Dict:
        n_joints = self.robot.get_num_joints()
        return {
            'left_gripper':    3,
            'right_gripper':   3,
            'rope':            FloatingRopeScenario.n_links * 3,
            'joint_positions': n_joints,
            'rgbd':            self.IMAGE_H * self.IMAGE_W * 4,
        }

    def observations_description(self) -> Dict:
        return {
            'left_gripper':  3,
            'right_gripper': 3,
            'rgbd':          self.IMAGE_H * self.IMAGE_W * 4,
        }

    def plot_state_rviz(self, state: Dict, **kwargs):
        FloatingRopeScenario.plot_state_rviz(self, state, **kwargs)
        label = kwargs.pop("label", "")
        # FIXME: the ACOs are part of the "environment", but they are needed to plot the state. leaky abstraction :(
        #  perhaps make them part of state_metadata?
        aco = state.get('attached_collision_objects', None)

        if 'joint_positions' in state and 'joint_names' in state:
            joint_state = joint_state_msg_from_state_dict(state)
            robot_state = RobotState(joint_state=joint_state, attached_collision_objects=aco)
            self.robot.display_robot_state(robot_state, label, kwargs.get("color", None))
        if add_predicted('joint_positions') in state and 'joint_names' in state:
            joint_state = joint_state_msg_from_state_dict_predicted(state)
            robot_state = RobotState(joint_state=joint_state, attached_collision_objects=aco)
            self.robot.display_robot_state(robot_state, label, kwargs.get("color", None))
        elif 'joint_positions' not in state:
            rospy.logwarn_throttle(10, 'no joint positions in state', logger_name=Path(__file__).stem)
        elif 'joint_names' not in state:
            rospy.logwarn_throttle(10, 'no joint names in state', logger_name=Path(__file__).stem)

    def dynamics_dataset_metadata(self):
        metadata = FloatingRopeScenario.dynamics_dataset_metadata(self)
        return metadata

    @staticmethod
    def simple_name():
        return "dual_arm"

    def get_excluded_models_for_env(self):
        exclude = ExcludeModelsRequest()
        res: ExcludeModelsResponse = self.exclude_from_planning_scene_srv(exclude)
        return res.all_model_names

    def initial_obstacle_poses_with_noise(self, env_rng: np.random.RandomState, obstacles: List):
        raise NotImplementedError()

    def get_environment(self, params: Dict, **kwargs):
        default_res = 0.01
        if 'res' not in params:
            rospy.logwarn(f"res not in params, using default {default_res}", logger_name=Path(__file__).stem)
            res = default_res
        else:
            res = params["res"]
        voxel_grid_env = get_environment_for_extents_3d(extent=params['extent'],
                                                        res=res,
                                                        service_provider=self.service_provider,
                                                        excluded_models=self.get_excluded_models_for_env())

        env = {}
        env.update({k: np.array(v).astype(np.float32) for k, v in voxel_grid_env.items()})
        env.update(MoveitPlanningSceneScenarioMixin.get_environment(self))

        return env

    @staticmethod
    def robot_name():
        raise NotImplementedError()

    def reset_cdcpd(self):
        # since the launch file has respawn=true, we just need to kill cdcpd_node and it will restart
        rosnode.kill_nodes("cdcpd_node")

    def needs_reset(self, state: Dict, params: Dict):
        grippers_out_of_bounds = self.grippers_out_of_bounds(state['left_gripper'], state['right_gripper'], params)
        return FloatingRopeScenario.needs_reset(self, state, params) or grippers_out_of_bounds

    def get_preferred_tool_orientations(self, tool_names: List[str]):
        """
        The purpose of this function it to make sure the tool orientations are in the order of tool_names
        Args:
            tool_names:

        Returns:

        """
        preferred_tool_orientations = []
        for tool_name in tool_names:
            if 'left' in tool_name:
                preferred_tool_orientations.append(self.left_preferred_tool_orientation)
            elif 'right' in tool_name:
                preferred_tool_orientations.append(self.right_preferred_tool_orientation)
            else:
                raise NotImplementedError()
        return preferred_tool_orientations

    def is_moveit_robot_in_collision(self, environment: Dict, state: Dict, action: Dict):
        scene: PlanningScene = environment['scene_msg']
        joint_state = joint_state_msg_from_state_dict(state)
        scene, robot_state = merge_joint_state_and_scene_msg(scene, joint_state)
        in_collision = self.robot.jacobian_follower.check_collision(scene, robot_state)
        return in_collision

    def moveit_robot_reached(self, state: Dict, action: Dict, next_state: Dict):
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        predicted_robot_state = robot_state_msg_from_state_dict(next_state)
        desired_tool_positions = [action['left_gripper_position'], action['right_gripper_position']]
        pred_tool_positions = self.robot.jacobian_follower.get_tool_positions(tool_names, predicted_robot_state)
        for pred_tool_position, desired_tool_position in zip(pred_tool_positions, desired_tool_positions):
            reached = np.allclose(desired_tool_position, pred_tool_position, atol=5e-3)
            if not reached:
                return False
        return True

    def follow_jacobian_from_example(self, example: Dict):
        j = self.robot.jacobian_follower
        batch_size = example["batch_size"]
        deserialize_scene_msg(example)
        scene_msg = example['scene_msg']
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        preferred_tool_orientations = self.get_preferred_tool_orientations(tool_names)
        target_reached_batched = []
        pred_joint_positions_batched = []
        joint_names_batched = []
        for b in range(batch_size):
            scene_msg_b: PlanningScene = scene_msg[b]
            input_sequence_length = example['left_gripper_position'].shape[1]
            target_reached = [True]
            pred_joint_positions = [example['joint_positions'][b, 0]]
            pred_joint_positions_t = example['joint_positions'][b, 0]
            joint_names_t = example['joint_names'][b, 0]
            joint_names = [joint_names_t]
            for t in range(input_sequence_length):
                left_gripper_points = [example['left_gripper_position'][b, t]]
                right_gripper_points = [example['right_gripper_position'][b, t]]
                grippers = [left_gripper_points, right_gripper_points]

                joint_state_b_t = make_joint_state(pred_joint_positions_t, to_list_of_strings(joint_names_t))
                scene_msg_b, robot_state = merge_joint_state_and_scene_msg(scene_msg_b, joint_state_b_t)
                plan: RobotTrajectory
                reached_t: bool
                plan, reached_t = j.plan(group_name='both_arms',
                                         tool_names=tool_names,
                                         preferred_tool_orientations=preferred_tool_orientations,
                                         start_state=robot_state,
                                         scene=scene_msg_b,
                                         grippers=grippers,
                                         max_velocity_scaling_factor=0.1,
                                         max_acceleration_scaling_factor=0.1)
                pred_joint_positions_t = get_joint_positions_given_state_and_plan(plan, robot_state)

                target_reached.append(reached_t)
                pred_joint_positions.append(pred_joint_positions_t)
                joint_names.append(joint_names_t)
            target_reached_batched.append(target_reached)
            pred_joint_positions_batched.append(pred_joint_positions)
            joint_names_batched.append(joint_names)

        pred_joint_positions_batched = np.array(pred_joint_positions_batched)
        target_reached_batched = np.array(target_reached_batched)
        joint_names_batched = np.array(joint_names_batched)
        return target_reached_batched, pred_joint_positions_batched, joint_names_batched

    def uniform_state_augmentation(self,
                                   input_dict: Dict,
                                   batch_size,
                                   time,
                                   seed: tfp.util.SeedStream):
        assert time == 2

        # apply those to the rope and grippers
        rope_points = tf.reshape(input_dict[add_predicted('rope')], [batch_size, time, -1, 3])
        left_gripper_point = input_dict[add_predicted('left_gripper')]
        right_gripper_point = input_dict[add_predicted('right_gripper')]
        left_gripper_points = tf.expand_dims(left_gripper_point, axis=-2)
        right_gripper_points = tf.expand_dims(right_gripper_point, axis=-2)

        # sample a new delta x,y,z,theta
        # TODO: implement rotation about z
        zeros = np.zeros_like(left_gripper_point[:, 0])
        delta_distribution = tfp.distributions.TruncatedNormal(zeros, 0.2, -0.5, 0.5)  # these are hyper-parameters
        theta_low = [-np.pi] * left_gripper_point.shape[0]
        theta_high = [np.pi] * left_gripper_point.shape[0]
        theta_distribution = tfp.distributions.Uniform(theta_low, theta_high)
        delta_position = delta_distribution.sample(seed=seed())
        theta = theta_distribution.sample(seed=seed())

        rotation_matrix = make_rotation_matrix_like(delta_position, theta)

        # rotates about the world origin, which isn't great because it's less likely to produce a feasible augmentation
        def _rot(points):
            n = points.shape[2]
            rotation_matrix_tiled = tf.tile(rotation_matrix[:, tf.newaxis, tf.newaxis], [1, 2, n, 1, 1])
            points_rotated = rotate_points_3d(rotation_matrix_tiled, points)
            return points_rotated

        rope_points_rotated = _rot(rope_points)
        left_gripper_points_rotated = _rot(left_gripper_points)
        right_gripper_points_rotated = _rot(right_gripper_points)

        delta_position = tf.tile(delta_position[:, tf.newaxis, tf.newaxis], [1, 2, 1, 1])

        left_gripper_points_aug = left_gripper_points_rotated + delta_position
        right_gripper_points_aug = right_gripper_points_rotated + delta_position
        rope_points_aug = rope_points_rotated + delta_position

        # compute the new action
        left_gripper_position = input_dict['left_gripper_position']
        right_gripper_position = input_dict['right_gripper_position']
        left_gripper_position_rotated = rotate_points_3d(rotation_matrix[:, tf.newaxis], left_gripper_position)
        right_gripper_position_rotated = rotate_points_3d(rotation_matrix[:, tf.newaxis], right_gripper_position)
        left_gripper_position_aug = left_gripper_position_rotated + delta_position[:, 0]
        right_gripper_position_aug = right_gripper_position_rotated + delta_position[:, 0]

        # use IK to get a new starting joint configuration
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        empty_scene_msgs = _deserialize_scene_msg(input_dict)
        for s in empty_scene_msgs:
            s.world.collision_objects = []

        out_joint_positions_start = []
        out_joint_positions_end = []
        reached = []
        for b in range(batch_size):
            # use the joint config pre-augmentation to see IK for the augmented joint config
            seed_joint_position_b = input_dict[add_predicted('joint_positions')][b, 0].numpy().tolist()
            joint_names = input_dict['joint_names'][b, 0].numpy().tolist()
            preferred_tool_orientations = self.get_preferred_tool_orientations(tool_names)

            left_gripper_aug_start_point = left_gripper_points_aug[b, 0, 0].numpy()
            right_gripper_aug_start_point = right_gripper_points_aug[b, 0, 0].numpy()
            left_gripper_aug_end_point = left_gripper_points_aug[b, 1, 0].numpy()
            right_gripper_aug_end_point = right_gripper_points_aug[b, 1, 0].numpy()
            grippers_start = [[left_gripper_aug_start_point], [right_gripper_aug_start_point]]
            grippers_end = [[left_gripper_aug_end_point], [right_gripper_aug_end_point]]

            # run jacobian follower as a hack to try to solve for a joint config with grippers matching the augmented
            # gripper positions
            seed_joint_state = JointState(name=joint_names, position=seed_joint_position_b)
            empty_scene_msg_b, seed_robot_state = merge_joint_state_and_scene_msg(empty_scene_msgs[b], seed_joint_state)
            plan_to_start, reached_start_b = self.robot.jacobian_follower.plan(
                group_name='whole_body',
                tool_names=tool_names,
                preferred_tool_orientations=preferred_tool_orientations,
                start_state=seed_robot_state,
                scene=empty_scene_msg_b,
                grippers=grippers_start,
                max_velocity_scaling_factor=0.1,
                max_acceleration_scaling_factor=0.1,
            )

            planned_to_start_points = plan_to_start.joint_trajectory.points
            if len(planned_to_start_points) > 0:
                out_joint_position_start_b = planned_to_start_points[-1].positions
            else:
                out_joint_position_start_b = seed_joint_position_b

            # run jacobian follower (again) to produce the next joint state and confirm the motion is feasible
            start_joint_state = JointState(name=joint_names, position=out_joint_position_start_b)
            empty_scene_msg_b, start_robot_state = merge_joint_state_and_scene_msg(empty_scene_msgs[b],
                                                                                   start_joint_state)
            plan_to_end, reached_end_b = self.robot.jacobian_follower.plan(
                group_name='whole_body',
                tool_names=tool_names,
                preferred_tool_orientations=preferred_tool_orientations,
                start_state=start_robot_state,
                scene=empty_scene_msg_b,
                grippers=grippers_end,
                max_velocity_scaling_factor=0.1,
                max_acceleration_scaling_factor=0.1,
            )

            planned_to_end_points = plan_to_end.joint_trajectory.points
            if len(planned_to_end_points) > 0:
                out_joint_position_end_b = planned_to_end_points[-1].positions
            else:
                out_joint_position_end_b = seed_joint_position_b

            reached_b = reached_start_b and reached_end_b

            out_joint_positions_start.append(out_joint_position_start_b)
            out_joint_positions_end.append(out_joint_position_end_b)
            reached.append(reached_b)

        joint_positions_aug = tf.stack((tf.constant(out_joint_positions_start, tf.float32),
                                        tf.constant(out_joint_positions_end, tf.float32)), axis=1)
        rope_aug = tf.reshape(rope_points_aug, [batch_size, time, -1])
        left_gripper_aug = tf.reshape(left_gripper_points_aug, [batch_size, time, -1])
        right_gripper_aug = tf.reshape(right_gripper_points_aug, [batch_size, time, -1])

        reached = tf.cast(reached, tf.float32)
        aug_valid = reached  # NOTE: there could be other constraints we need to check?
        aug_valid_expanded = aug_valid[:, tf.newaxis, tf.newaxis]

        update_if_valid(input_dict, aug_valid_expanded, add_predicted('joint_positions'), joint_positions_aug)
        update_if_valid(input_dict, aug_valid_expanded, add_predicted('rope'), rope_aug)
        update_if_valid(input_dict, aug_valid_expanded, add_predicted('left_gripper'), left_gripper_aug)
        update_if_valid(input_dict, aug_valid_expanded, add_predicted('right_gripper'), right_gripper_aug)
        update_if_valid(input_dict, aug_valid_expanded, 'left_gripper_position', left_gripper_position_aug)
        update_if_valid(input_dict, aug_valid_expanded, 'right_gripper_position', right_gripper_position_aug)
        # update_if_valid(input_dict, aug_valid_expanded, 'env', env_aug) ????
        # or we could return?

        if DEBUG_VIZ_STATE_AUG:
            stepper = RvizSimpleStepper()
            for b in debug_viz_batch_indices(batch_size):
                env_b = {
                    'env':    input_dict['env'][b],
                    'res':    input_dict['res'][b],
                    'extent': input_dict['extent'][b],
                    'origin': input_dict['origin'][b],
                }

                self.plot_environment_rviz(env_b)
                self.debug_viz_state_action(input_dict, b, 'aug', color='white')

                stepper.step()

    def debug_viz_state_action(self, input_dict, b, label: str, color='red'):
        state_keys = ['left_gripper', 'right_gripper', 'rope']
        action_keys = ['left_gripper_position', 'right_gripper_position']
        state_0 = numpify({k: input_dict[add_predicted(k)][b, 0] for k in state_keys})
        action_0 = numpify({k: input_dict[k][b, 0] for k in action_keys})
        state_1 = numpify({k: input_dict[add_predicted(k)][b, 1] for k in state_keys})
        error_msg = Float32()
        error_t = input_dict['error'][b, 1]
        error_msg.data = error_t
        self.plot_state_rviz(state_0, idx=0, label=label, color=color)
        self.plot_state_rviz(state_1, idx=1, label=label, color=color)
        self.plot_action_rviz(state_0, action_0, idx=1, label=label, color=color)
        self.plot_is_close(input_dict['is_close'][b, 1])
        self.error_pub.publish(error_msg)


def update_if_valid(d: Dict, is_valid, k: str, v_aug):
    d[k] = is_valid * v_aug + (1 - is_valid) * d[k]
