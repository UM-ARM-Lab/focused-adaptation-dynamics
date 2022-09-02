from copy import deepcopy
from time import sleep
from typing import Dict, Optional

import numpy as np
import open3d
import rospkg
from pyrope_reset_planner import RopeResetPlanner, PlanningResult

import ros_numpy
import rospy
from arm_robots.cartesian import pose_distance
from arm_robots.robot import RobotPlanningError
from geometry_msgs.msg import Pose, Quaternion
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.dual_arm_rope_action import dual_arm_rope_execute_action
from link_bot_pycommon.get_cdcpd_state import GetCdcpdState
from link_bot_pycommon.get_joint_state import GetJointState
from link_bot_pycommon.grid_utils_np import extent_res_to_origin_point, extent_to_env_shape
from link_bot_pycommon.moveit_planning_scene_mixin import MoveitPlanningSceneScenarioMixin
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetMotionPlan
from tf.transformations import quaternion_from_euler


def wiggle_positions(current, n, s=0.02):
    rng = np.random.RandomState(0)
    for i in range(n):
        delta = rng.uniform([-s, -s, -s], [s, s, s])
        yield current + delta


class DualArmRealValRopeScenario(BaseDualArmRopeScenario):
    real = True

    COLOR_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_color_rect"
    DEPTH_IMAGE_TOPIC = "/kinect2_tripodA/qhd/image_depth_rect"

    def __init__(self, params: Optional[dict] = None):
        super().__init__('hdt_michigan', params)
        self.fast = False
        self.left_preferred_tool_orientation = quaternion_from_euler(-1.801, -1.141, -0.335)
        self.right_preferred_tool_orientation = quaternion_from_euler(-2.309, -1.040, 1.251)

        self.get_joint_state = GetJointState(self.robot)
        self.get_cdcpd_state = GetCdcpdState(self.tf, self.root_link)

        self.reset_move_group = 'both_arms'

        self.plan_srv = rospy.ServiceProxy("/hdt_michigan/plan_kinematic_path", GetMotionPlan)

    def execute_action(self, environment, state, action: Dict):
        action_fk = self.action_relative_to_fk(action, state)
        try:
            dual_arm_rope_execute_action(self, self.robot, environment, state, action_fk, vel_scaling=1.0,
                                         check_overstretching=False)
        except RuntimeError as e:
            print(e)

        if not self.fast:
            # NOTE sleeping because CDCPD is laggy.
            #  We sleep here instead of in get_state because we only need to sleep after we've moved
            rospy.sleep(4)

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
        super().on_before_data_collection(params)  # FIXME: this method could use the automatic procedure?

        joint_names = self.robot.get_joint_names('both_arms')
        current_joint_positions = np.array(self.robot.get_joint_positions(joint_names))
        reset_joint_config = np.array([params['reset_joint_config'][n] for n in joint_names])
        near_start = np.max(np.abs(reset_joint_config - current_joint_positions)) < 0.02
        grippers_are_closed = self.robot.is_left_gripper_closed() and self.robot.is_right_gripper_closed()
        if not near_start or not grippers_are_closed:
            # let go
            # self.robot.open_left_gripper()

            # move to init positions
            self.robot.plan_to_joint_config("both_arms", reset_joint_config.tolist())

            self.robot.speak("press enter to close grippers")
            print("Use the gamepad to close the left gripper")

        self.robot.speak("press enter to begin")
        while True:
            k = input("Ready to begin? [y]")
            if k in ['y', 'Y']:
                break
        print("Done.")

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
        left_start_pose.position.y = 0.48
        left_start_pose.position.z = 0.6
        right_start_pose = deepcopy(left_start_pose)
        right_start_pose.position.x = 0.2
        right_start_pose.orientation = ros_numpy.msgify(Quaternion, self.right_preferred_tool_orientation)

        right_tool_grasp_pose = Pose()
        right_tool_grasp_pose.position.x = 0.1
        right_tool_grasp_pose.position.y = 0.3
        right_tool_grasp_pose.position.z = 0.925
        right_tool_grasp_orientation = quaternion_from_euler(-1.5707, -1.5707, 0.6)
        right_tool_grasp_pose.orientation = ros_numpy.msgify(Quaternion, right_tool_grasp_orientation)
        self.tf.send_transform_from_pose_msg(right_tool_grasp_pose, 'robot_root', 'right_grasp')

        left_tool_grasp_pose = deepcopy(right_tool_grasp_pose)
        left_tool_grasp_pose.position.z = right_tool_grasp_pose.position.z - 0.835
        left_tool_grasp_pose.position.y += 0.01
        left_tool_grasp_pose.orientation = ros_numpy.msgify(Quaternion,
                                                            quaternion_from_euler(0, np.pi / 2 + 0.2, 0))

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

        print("Planning to grasp")
        try:
            # self.tf.send_transform_from_pose_msg(left_tool_grasp_pose, 'robot_root', 'left_tool_grasp_pose')
            plan_to_grasp(left_tool_grasp_pose, right_tool_grasp_pose, rrp, self.robot)

            # wait for rope to stop swinging
            print("Waiting for rope to settle")
            sleep(12)

            self.robot.store_current_tool_orientations(both_tools)
            left_up = ros_numpy.numpify(left_tool_grasp_pose.position) + np.array([0, 0.05, .25])
            right_grasp_position_np = ros_numpy.numpify(right_tool_grasp_pose.position)
            self.robot.follow_jacobian_to_position('both_arms', both_tools, [[left_up], [right_grasp_position_np]])

            left_arm_joint_names = self.robot.get_joint_names('left_arm')
            left_flip_config = self.robot.get_joint_positions(left_arm_joint_names)
            left_flip_config[-1] -= np.pi
            self.robot.plan_to_joint_config('left_arm', left_flip_config)

            print("Planning to start")
            plan_to_start(left_start_pose, right_start_pose, rrp, self.robot)
        except RobotPlanningError:
            # at this point give up and ask peter to fix the rope
            self.robot.plan_to_poses('both_arms', both_tools, [left_start_pose, right_start_pose])
            input("Please fix the rope!")

        # restore
        self.robot.store_tool_orientations({
            'left_tool':  self.left_preferred_tool_orientation,
            'right_tool': self.right_preferred_tool_orientation,
        })
        print("done.")

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        raise NotImplementedError()

    def needs_reset(self, state: Dict, params: Dict):
        # FIXME:
        return False

    def on_after_data_collection(self, params):
        self.robot.disconnect()

    def grasp_rope_endpoints(self, *args, **kwargs):
        pass

    def get_environment(self, params: Dict, **kwargs):
        default_res = 0.02
        if 'res' not in params:
            rospy.logwarn(f"res not in params, using default {default_res}", logger_name=Path(__file__).stem)
            res = default_res
        else:
            res = params["res"]

        env = {}

        # voxel_grid_env = get_environment_for_extents_3d(extent=params['extent'],  # TODO: use merged_points?
        #                                                 res=res,
        #                                                 frame='robot_root',
        #                                                 service_provider=self.service_provider,
        #                                                 excluded_models=self.get_excluded_models_for_env())
        # env.update({k: np.array(v).astype(np.float32) for k, v in voxel_grid_env.items()})

        r = rospkg.RosPack()
        perception_pkg_dir = r.get_path('link_bot_perception')
        pcd = open3d.io.read_point_cloud(perception_pkg_dir + "/pcd_files/real_car_env_for_mde.pcd")
        points = np.asarray(pcd.points)

        extent = params['extent']
        origin_point = extent_res_to_origin_point(extent, res)
        shape = extent_to_env_shape(extent, res)
        vg = np.zeros(shape, dtype=np.float32)
        indices = ((points - origin_point) / res).astype(np.int64)
        in_bounds_lower = np.all(indices > 0, axis=1)
        in_bounds_upper = np.all(indices < shape, axis=1)
        which_indices_are_valid = np.where(np.logical_and(in_bounds_lower, in_bounds_upper))[0]
        valid_indices = indices[which_indices_are_valid]
        rows = valid_indices[:, 1]
        cols = valid_indices[:, 0]
        channels = valid_indices[:, 2]
        vg[rows, cols, channels] = 1.0

        env['env'] = vg
        env['extent'] = extent
        env['res'] = res
        env['origin_point'] = origin_point

        # self.plot_points_rviz(points, label='env_points')
        # self.plot_environment_rviz(env)

        env.update(MoveitPlanningSceneScenarioMixin.get_environment(self))

        return env


def plan_to_start(left_start_pose, right_start_pose, rrp, val):
    pub = rospy.Publisher("/test_rope_reset_planner/ompl_plan", DisplayTrajectory, queue_size=10)

    orientation_path_tol = 0.4
    while True:
        result: PlanningResult = rrp.plan_to_start(left_start_pose, right_start_pose, max_gripper_distance=0.73,
                                                   orientation_path_tolerance=orientation_path_tol,
                                                   orientation_goal_tolerance=0.2,
                                                   timeout=60)
        print(result.status)

        if result.status == "Exact solution":
            break

        orientation_path_tol += 0.1

        if orientation_path_tol >= 1.5:
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
    orientation_path_tol = 0.4

    while True:
        result = rrp.plan_to_reset(left_tool_grasp_pose, right_tool_grasp_pose, orientation_path_tol, 0.3, 60)
        if result.status == 'Exact solution':
            break

        orientation_path_tol += 0.1

        if orientation_path_tol >= 1.5:
            raise RobotPlanningError("could not plan to grasp!")

    display_msg = DisplayTrajectory()
    display_msg.trajectory.append(result.traj)

    for _ in range(5):
        rospy.sleep(0.1)
        pub.publish(display_msg)

    result.traj.joint_trajectory.header.stamp = rospy.Time.now()
    val.follow_arms_joint_trajectory(result.traj.joint_trajectory)
