import pathlib
from copy import deepcopy
from time import sleep
from typing import Dict, Optional

import numpy as np
import rospkg
from colorama import Fore
from pyjacobian_follower import JacobianFollower
from pyrope_reset_planner import RopeResetPlanner, PlanningResult

import ros_numpy
import rospy
from arm_robots.cartesian import pose_distance
from arm_robots.robot import RobotPlanningError
from arm_robots.robot_utils import merge_joint_state_and_scene_msg
from geometry_msgs.msg import Pose, Quaternion
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario, \
    get_joint_positions_given_state_and_plan
from link_bot_pycommon.dual_arm_rope_action import dual_arm_rope_execute_action
from link_bot_pycommon.get_cdcpd_state import GetCdcpdState
from link_bot_pycommon.get_joint_state import GetJointState
from link_bot_pycommon.grid_utils_np import extent_res_to_origin_point, extent_to_env_shape
from link_bot_pycommon.moveit_planning_scene_mixin import MoveitPlanningSceneScenarioMixin
from link_bot_pycommon.moveit_utils import make_joint_state
from moonshine.tensorflow_utils import to_list_of_strings
from moveit_msgs.msg import DisplayTrajectory, PlanningScene, RobotTrajectory
from moveit_msgs.srv import GetMotionPlan
from tf.transformations import quaternion_from_euler

planning_scene_scale = 1.0
execution_scene_scale = 1.0


def wiggle_positions(current, n, s=0.02):
    rng = np.random.RandomState(0)
    for i in range(n):
        delta = rng.uniform([-s, -s, -s], [s, s, s])
        yield current + delta


class DualArmRealValRopeScenario(BaseDualArmRopeScenario):
    real = True

    def __init__(self, params: Optional[dict] = None):
        super().__init__('hdt_michigan', params)
        self.fast = False
        self.padded_scene_ = None
        self.left_preferred_tool_orientation = quaternion_from_euler(-1.801, -1.141, -0.335)
        self.right_preferred_tool_orientation = quaternion_from_euler(-2.309, -1.040, 1.251)

        self.get_joint_state = GetJointState(self.robot)
        self.get_cdcpd_state = GetCdcpdState(self.tf, self.root_link)

        self.reset_move_group = 'both_arms'

        from cdcpd.srv import SetGripperConstraints
        self.plan_srv = rospy.ServiceProxy("/hdt_michigan/plan_kinematic_path", GetMotionPlan)
        self.cdcpd_constraint_srv = rospy.ServiceProxy("cdcpd_node/set_gripper_constraints", SetGripperConstraints)

    def execute_action(self, environment, state, action: Dict):
        action_fk = self.action_relative_to_fk(action, state)
        dual_arm_rope_execute_action(self, self.robot, environment, state, action_fk, vel_scaling=1.0,
                                     check_overstretching=False)

    def action_relative_to_fk(self, action, state):
        robot_state = self.get_robot_state.get_state()
        # so state gets the gripper positions via the mocap markers
        left_gripper_position_mocap = state['left_gripper']
        right_gripper_position_mocap = state['right_gripper']
        left_gripper_delta_position = action['left_gripper_position'] - left_gripper_position_mocap
        # whereas this is via fk
        left_gripper_position_fk = robot_state['left_gripper']
        right_gripper_delta_position = action['right_gripper_position'] - right_gripper_position_mocap
        right_gripper_position_fk = robot_state['right_gripper']
        action_fk = {
            'left_gripper_position':  left_gripper_position_fk + left_gripper_delta_position,
            'right_gripper_position': right_gripper_position_fk + right_gripper_delta_position,
        }
        self.tf.send_transform(action_fk['left_gripper_position'], [0, 0, 0, 1], parent=self.root_link,
                               child='left_gripper_position_fk', is_static=True)
        self.tf.send_transform(action_fk['right_gripper_position'], [0, 0, 0, 1], parent=self.root_link,
                               child='right_gripper_position_fk', is_static=True)
        return action_fk

    def on_before_data_collection(self, params: Dict):
        super().on_before_data_collection(params)

    def get_state(self):
        state = {}
        state.update(self.get_robot_state.get_state())
        state.update(self.get_cdcpd_state.get_state())
        # I'm pretty sure that specifying time as now() is necessary to ensure we get the absolute latest transform
        left_gripper_mocap = "mocap_left_hand_left_hand"
        right_gripper_mocap = "mocap_right_hand_right_hand"
        state['left_gripper'] = self.tf.get_transform(self.root_link, left_gripper_mocap)[:3, 3]
        state['right_gripper'] = self.tf.get_transform(self.root_link, right_gripper_mocap)[:3, 3]

        return state

    def get_excluded_models_for_env(self):
        return []

    def reset_to_start(self, planner_params: Dict, start: Dict):
        self.fix_oob_joints()

        self.robot.store_tool_orientations({
            'left_tool':  self.left_preferred_tool_orientation,
            'right_tool': self.right_preferred_tool_orientation,
        })

        left_tool_post_planning_pose = Pose()
        left_tool_post_planning_pose.orientation = ros_numpy.msgify(Quaternion, self.left_preferred_tool_orientation)
        left_tool_post_planning_pose.position.x = -0.2
        left_tool_post_planning_pose.position.y = 0.6
        left_tool_post_planning_pose.position.z = 0.34
        right_tool_post_planning_pose = deepcopy(left_tool_post_planning_pose)
        right_tool_post_planning_pose.orientation = ros_numpy.msgify(Quaternion, self.right_preferred_tool_orientation)
        right_tool_post_planning_pose.position.x = 0.2

        left_start_pose = Pose()
        left_start_pose.orientation = ros_numpy.msgify(Quaternion, self.left_preferred_tool_orientation)
        left_start_pose.position.x = -0.2
        left_start_pose.position.y = 0.55
        left_start_pose.position.z = 0.65
        right_start_pose = deepcopy(left_start_pose)
        right_start_pose.position.x = 0.2
        right_start_pose.orientation = ros_numpy.msgify(Quaternion, self.right_preferred_tool_orientation)

        right_tool_grasp_pose = Pose()
        right_tool_grasp_pose.position.x = 0.18
        right_tool_grasp_pose.position.y = 0.33
        right_tool_grasp_pose.position.z = 1.04
        right_tool_grasp_orientation = quaternion_from_euler(-2.2, -2.2, 1.0)
        right_tool_grasp_pose.orientation = ros_numpy.msgify(Quaternion, right_tool_grasp_orientation)
        self.tf.send_transform_from_pose_msg(right_tool_grasp_pose, 'robot_root', 'right_grasp')
        right_grasp_position_np = ros_numpy.numpify(right_tool_grasp_pose.position)

        left_tool_grasp_pose = deepcopy(right_tool_grasp_pose)
        left_tool_grasp_pose.position.z = right_tool_grasp_pose.position.z - 0.89
        left_tool_grasp_pose.orientation = ros_numpy.msgify(Quaternion,
                                                            quaternion_from_euler(0, np.pi / 2 + 0.2, 0))
        # self.tf.send_transform_from_pose_msg(left_tool_grasp_pose, 'robot_root', 'left_tool_grasp_pose')

        initial_left_pose = self.robot.get_link_pose("left_tool")
        initial_right_pose = self.robot.get_link_pose("right_tool")
        left_pose_error = pose_distance(left_start_pose, initial_left_pose)
        right_pose_error = pose_distance(right_start_pose, initial_right_pose)
        if left_pose_error < 0.05 and right_pose_error < 0.05:
            q = input("Already at start! Should I skip the reset? [N/y] ")
            if q in ['Y', 'y']:
                print("Skipping reset")
                return
            else:
                print("Doing reset anyways")

        both_tools = ['left_tool', 'right_tool']

        rrp = RopeResetPlanner()

        self.set_cdcpd_both()

        succeeded = False
        for i in range(10):
            try:
                # first move up a bit in case the torso is in collision?
                self.move_torso_back()

                print("Planning to grasp")
                plan_to_grasp(left_tool_grasp_pose, right_tool_grasp_pose, rrp, self.robot)
                self.robot.store_current_tool_orientations(both_tools)

                # wait for rope to stop swinging
                print("Waiting for rope to settle")
                sleep(30)

                # change CDCPD constraints
                self.set_cdcpd_right_only()

                sleep(5)

                # servo to the rope?
                cdcpd_state = self.get_cdcpd_state.get_state()
                left_at_rope = cdcpd_state['rope'].reshape([25, 3])[0]
                left_at_rope[2] = left_tool_grasp_pose.position.z
                self.robot.follow_jacobian_to_position('both_arms', both_tools, [[left_at_rope], [right_grasp_position_np]])

                sleep(5)

                left_up = left_at_rope + np.array([0, 0, .2])
                right_down = right_grasp_position_np + np.array([0, 0, -0.08])
                _bak = self.robot.called.jacobian_target_not_reached_is_failure
                self.robot.called.jacobian_target_not_reached_is_failure = False
                self.robot.follow_jacobian_to_position('both_arms', both_tools, [[left_up], [right_down]])

                self.set_cdcpd_both()

                self.flip_left_wrist()

                self.robot.store_current_tool_orientations(both_tools)

                left_up_out = left_at_rope + np.array([-0.08, 0, .18])
                right_down_out = right_grasp_position_np + np.array([0.08, 0, -0.1])
                self.robot.follow_jacobian_to_position('both_arms', both_tools, [[left_up_out], [right_down_out]])
                self.robot.called.jacobian_target_not_reached_is_failure = _bak

                try:
                    print("Planning to start using jacobian follower")
                    self.plan_to_start_with_jacobian_follower(both_tools, left_start_pose, right_start_pose)
                except RobotPlanningError:
                    print("Planning to start")
                    plan_to_start(left_start_pose, right_start_pose, rrp, self.robot)

                # Now remove the constraint and see if it stays where it is. That's a sign the reset worked
                self.set_cdcpd_right_only()
                sleep(30)

                cdcpd_state = self.get_cdcpd_state.get_state()
                left_at_rope = cdcpd_state['rope'].reshape([25, 3])[0]
                expected_left_at_rope = self.tf.get_transform(self.root_link, "mocap_left_hand_left_hand")[:3, 3]
                rope_in_hand = np.linalg.norm(left_at_rope - expected_left_at_rope) < 0.12
                if rope_in_hand:
                    # change CDCPD constraints
                    self.set_cdcpd_both()
                    succeeded = True
                    break
            except RobotPlanningError:
                print("Trying again!")
                pass
                # # at this point give up and ask peter to fix the rope
                # self.robot.plan_to_poses('both_arms', both_tools, [left_start_pose, right_start_pose])
                # input("Please fix the rope!")

        if not succeeded:
            raise RuntimeError("Failed to plan to start after 10 attempts!")

        # restore
        self.robot.store_tool_orientations({
            'left_tool':  self.left_preferred_tool_orientation,
            'right_tool': self.right_preferred_tool_orientation,
        })
        print("done.")

    def plan_to_start_with_jacobian_follower(self, both_tools, left_start_pose, right_start_pose):
        self.robot.store_tool_orientations({
            'left_tool':  self.left_preferred_tool_orientation,
            'right_tool': self.right_preferred_tool_orientation,
        })
        self.robot.follow_jacobian_to_position('both_arms', both_tools,
                                               [[ros_numpy.numpify(left_start_pose.position)],
                                                [ros_numpy.numpify(right_start_pose.position)]])

    def fix_oob_joints(self):
        pass

    def flip_left_wrist(self):
        left_arm_joint_names = self.robot.get_joint_names('left_arm')
        left_flip_config = self.robot.get_joint_positions(left_arm_joint_names)
        if left_flip_config[-1] > 0:
            left_flip_config[-1] -= np.pi
        else:
            left_flip_config[-1] += np.pi
        self.robot.plan_to_joint_config('left_arm', left_flip_config)

    def move_torso_back(self):
        torso_joint_names = self.robot.get_joint_names('torso')
        back_config = self.robot.get_joint_positions(torso_joint_names)
        if back_config[1] > 0.2:
            back_config[1] -= 0.2
            self.robot.plan_to_joint_config('torso', back_config)

    def set_cdcpd_both(self):
        from cdcpd.msg import GripperConstraint
        from cdcpd.srv import SetGripperConstraintsRequest
        left_gripper_cdcpd_constraint = GripperConstraint()
        left_gripper_cdcpd_constraint.frame_id = "mocap_left_hand_left_hand"
        left_gripper_cdcpd_constraint.node_index = 0
        right_gripper_cdcpd_constraint = GripperConstraint()
        right_gripper_cdcpd_constraint.frame_id = "mocap_right_hand_right_hand"
        right_gripper_cdcpd_constraint.node_index = 24
        set_cdcpd_constraints = SetGripperConstraintsRequest()
        set_cdcpd_constraints.constraints.append(left_gripper_cdcpd_constraint)
        set_cdcpd_constraints.constraints.append(right_gripper_cdcpd_constraint)
        self.cdcpd_constraint_srv(set_cdcpd_constraints)

    def set_cdcpd_right_only(self):
        from cdcpd.msg import GripperConstraint
        from cdcpd.srv import SetGripperConstraintsRequest
        right_gripper_cdcpd_constraint = GripperConstraint()
        right_gripper_cdcpd_constraint.frame_id = "mocap_right_hand_right_hand"
        right_gripper_cdcpd_constraint.node_index = 24
        set_cdcpd_constraints = SetGripperConstraintsRequest()
        set_cdcpd_constraints.constraints.append(right_gripper_cdcpd_constraint)
        self.cdcpd_constraint_srv(set_cdcpd_constraints)

    def set_cdcpd_left_only(self):
        from cdcpd.msg import GripperConstraint
        from cdcpd.srv import SetGripperConstraintsRequest
        left_gripper_cdcpd_constraint = GripperConstraint()
        left_gripper_cdcpd_constraint.frame_id = "mocap_left_hand_left_hand"
        left_gripper_cdcpd_constraint.node_index = 0
        set_cdcpd_constraints = SetGripperConstraintsRequest()
        set_cdcpd_constraints.constraints.append(left_gripper_cdcpd_constraint)
        self.cdcpd_constraint_srv(set_cdcpd_constraints)

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        raise NotImplementedError()

    def needs_reset(self, state: Dict, params: Dict):
        return False

    def on_after_data_collection(self, params):
        self.robot.disconnect()

    def grasp_rope_endpoints(self, *args, **kwargs):
        pass

    def get_environment(self, params: Dict, **kwargs):
        default_res = 0.02
        if 'res' not in params:
            rospy.logwarn(f"res not in params, using default {default_res}", logger_name=pathlib.Path(__file__).stem)
            res = default_res
        else:
            res = params["res"]

        r = rospkg.RosPack()
        perception_pkg_dir = r.get_path('link_bot_perception')
        import open3d
        pcd = open3d.io.read_point_cloud(perception_pkg_dir + "/pcd_files/real_car_env_for_mde.pcd")
        points = np.asarray(pcd.points) + np.array([0.02, 0, 0.015])

        extent = params['extent']
        origin_point = extent_res_to_origin_point(extent, res)
        shape = extent_to_env_shape(extent, res)
        shape_yxz = np.array([shape[1], shape[0], shape[2]])
        vg = np.zeros(shape, dtype=np.float32)
        indices = ((points - origin_point) / res).astype(np.int64)
        in_bounds_lower = np.all(indices > 0, axis=1)
        in_bounds_upper = np.all(indices < shape_yxz, axis=1)
        which_indices_are_valid = np.where(np.logical_and(in_bounds_lower, in_bounds_upper))[0]
        valid_indices = indices[which_indices_are_valid]
        rows = valid_indices[:, 1]
        cols = valid_indices[:, 0]
        channels = valid_indices[:, 2]
        vg[rows, cols, channels] = 1.0

        env_pcd = {
            'env':          vg,
            'extent':       extent,
            'res':          res,
            'origin_point': origin_point
        }

        # self.plot_points_rviz(points, label='env_points')
        env = {k: np.array(v).astype(np.float32) for k, v in env_pcd.items()}
        self.plot_environment_rviz(env)

        env.update(MoveitPlanningSceneScenarioMixin.get_environment(self))

        print(Fore.RED + "Storing current planning scene and using it for all future planning!" + Fore.RESET)
        self.padded_scene_ = self.pad_robot_links(env['scene_msg'])
        self.robot.jacobian_follower.store_scene(self.padded_scene_)

        return env

    def follow_jacobian_from_example(self, example: Dict, j: Optional[JacobianFollower] = None):
        if j is None:
            j = self.robot.jacobian_follower
        batch_size = example["batch_size"]
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
                # Transform into the right frame
                left_gripper_point = example['left_gripper_position'][b, t]
                right_gripper_point = example['right_gripper_position'][b, t]
                grippers = [[left_gripper_point], [right_gripper_point]]

                joint_state_b_t = make_joint_state(pred_joint_positions_t, to_list_of_strings(joint_names_t))
                plan: RobotTrajectory
                reached_t: bool
                if self.padded_scene_ is None:
                    _, robot_state = merge_joint_state_and_scene_msg(scene_msg_b, joint_state_b_t)
                    plan, reached_t = j.plan(group_name='both_arms',
                                             tool_names=tool_names,
                                             preferred_tool_orientations=preferred_tool_orientations,
                                             scene=scene_msg_b,
                                             start_state=robot_state,
                                             grippers=grippers,
                                             max_velocity_scaling_factor=0.1,
                                             max_acceleration_scaling_factor=0.1)
                else:
                    _, robot_state = merge_joint_state_and_scene_msg(self.padded_scene_, joint_state_b_t)
                    plan, reached_t = j.plan_with_stored_scene(group_name='both_arms',
                                                               tool_names=tool_names,
                                                               preferred_tool_orientations=preferred_tool_orientations,
                                                               start_state=robot_state,
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

    def pad_collision_objects(self, scene_msg_b, padding=0.03):
        padded_scene_msg = deepcopy(scene_msg_b)
        for co in padded_scene_msg.world.collision_objects:
            for primitive in co.primitives:
                d = np.array(primitive.dimensions)
                primitive.dimensions = tuple((d + padding).tolist())
        return padded_scene_msg

    def pad_robot_links(self, scene_msg: PlanningScene):
        padded_scene_msg = deepcopy(scene_msg)
        links_to_pad = {
            'end_effector_right': 0.015,
            'end_effector_left':  0.015,
            'leftwrist':          0.02,
            'rightwrist':         0.02,
            'torso':              0.03,
            'leftforearm':        0.02,
            'rightforearm':       0.02,
        }
        for aco in padded_scene_msg.robot_state.attached_collision_objects:
            aco.object.primitives[0].dimensions = (0.07,)
        for link_padding in padded_scene_msg.link_padding:
            for link_name, padding in links_to_pad.items():
                if link_name == link_padding.link_name:
                    link_padding.padding = padding
        return padded_scene_msg


def plan_to_start(left_start_pose, right_start_pose, rrp, val):
    pub = rospy.Publisher("/test_rope_reset_planner/ompl_plan", DisplayTrajectory, queue_size=10)

    orientation_path_tol = 1.0
    while True:
        result: PlanningResult = rrp.plan_to_start(left_start_pose, right_start_pose, max_gripper_distance=0.715,
                                                   orientation_path_tolerance=orientation_path_tol,
                                                   orientation_goal_tolerance=0.2,
                                                   timeout=120, debug_collisions=False)

        if result.status == "Exact solution":
            break

        orientation_path_tol += 0.2

        if orientation_path_tol >= 2:
            raise RobotPlanningError("could not plan to start!")

    display_msg = DisplayTrajectory()
    display_msg.trajectory.append(result.traj)
    for _ in range(5):
        rospy.sleep(0.1)
        pub.publish(display_msg)

    result.traj.joint_trajectory.header.stamp = rospy.Time.now()
    val.follow_arms_joint_trajectory(result.traj.joint_trajectory)


def plan_to_grasp(left_tool_grasp_pose, right_tool_grasp_pose, rrp, val):
    pub = rospy.Publisher("/test_rope_reset_planner/ompl_plan", DisplayTrajectory, queue_size=10)
    orientation_path_tol = 0.6

    while True:
        result = rrp.plan_to_reset(left_tool_grasp_pose, right_tool_grasp_pose, orientation_path_tol, 0.3, timeout=120,
                                   debug_collisions=False)
        if result.status == 'Exact solution':
            break

        orientation_path_tol += 0.2

        if orientation_path_tol >= 2:
            raise RobotPlanningError("could not plan to grasp!")

    display_msg = DisplayTrajectory()
    display_msg.trajectory.append(result.traj)

    for _ in range(5):
        rospy.sleep(0.1)
        pub.publish(display_msg)

    result.traj.joint_trajectory.header.stamp = rospy.Time.now()
    val.follow_arms_joint_trajectory(result.traj.joint_trajectory)
