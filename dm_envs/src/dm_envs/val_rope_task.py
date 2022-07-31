from typing import Dict

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.utils import inverse_kinematics
from transformations import quaternion_from_euler

import rospy
from arc_utilities import ros_init
from arm_robots.robot_utils import interpolate_joint_trajectory_points, get_ordered_tolerance_list, is_waypoint_reached, \
    waypoint_error, make_follow_joint_trajectory_goal
from dm_envs.base_rope_task import BaseRopeManipulation
from dm_envs.mujoco_services import my_step
from dm_envs.mujoco_visualizer import MujocoVisualizer
from moveit_msgs.msg import RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class VoxelgridBuild(composer.Entity):
    def _build(self, res: float):
        self._model = mjcf.element.RootElement(model='vgb_sphere')
        self._geom = self._model.worldbody.add('geom', name='geom', type='sphere', size=[res])

    @property
    def mjcf_model(self):
        return self._model


class StaticEnvEntity(composer.Entity):
    def _build(self, path: str):
        print(f"Loading {path}")
        self._model = mjcf.from_path(path)

    @property
    def mjcf_model(self):
        return self._model


class ValEntity(composer.Entity):
    def _build(self):
        self._model = mjcf.from_path('val_husky_no_gripper_collisions.xml')

    @property
    def mjcf_model(self):
        return self._model

    @property
    def joints(self):
        return self.mjcf_model.find_all('joint')

    @property
    def joint_names(self):
        return [j.name for j in self.joints]


class ValRopeManipulation(BaseRopeManipulation):

    def __init__(self, params: Dict):
        super().__init__(params)

        # other entities
        self._val = ValEntity()
        self._static_env = StaticEnvEntity(params.get('static_env_filename', 'empty.xml'))
        self.vgb = VoxelgridBuild(res=0.01)

        val_site = self._arena.attach(self._val)
        val_site.pos = [0, 0, 0.15]
        static_env_site = self._arena.attach(self._static_env)
        static_env_site.pos = [1.22, -0.14, 0.1]
        static_env_site.quat = quaternion_from_euler(0, 0, -1.5707)
        self._arena.add_free_entity(self.vgb)

        self._arena.mjcf_model.equality.add('distance', name='left_grasp', geom1='val/left_tool_geom', geom2='rope/rG0',
                                            distance=0, active='false', solref="0.02 2")
        self._arena.mjcf_model.equality.add('distance', name='right_grasp', geom1='val/right_tool_geom',
                                            geom2=f'rope/rG{self.rope.length - 1}', distance=0, active='false',
                                            solref="0.02 2")

        self._actuators = self._arena.mjcf_model.find_all('actuator')

        self._task_observables.update({
            'left_gripper':    observable.MujocoFeature('site_xpos', 'val/left_tool'),
            'right_gripper':   observable.MujocoFeature('site_xpos', 'val/right_tool'),
            'joint_positions': observable.MJCFFeature('qpos', self.actuated_joints),
        })

        for obs_ in self._task_observables.values():
            obs_.enabled = True

    @property
    def joints(self):
        return self._val.joints

    @property
    def actuated_joints(self):
        return [a.joint for a in self._actuators]

    @property
    def joint_names(self):
        return [f'val/{n}' for n in self._val.joint_names]

    @property
    def actuated_joint_names(self):
        return [f'val/{a.joint.name}' for a in self._actuators]

    def initialize_episode(self, physics, random_state):
        with physics.reset_context():
            # this will overwrite the pose set when val is 'attach'ed to the arena
            self._val.set_pose(physics,
                               position=[0, 0, 0.15],
                               quaternion=quaternion_from_euler(0, 0, 0))
            self.rope.set_pose(physics,
                               position=[0.5, -self.rope.length_m / 2, 0.6],
                               quaternion=quaternion_from_euler(0, 0, 1.5707))
            for i in range(self.rope.length - 1):
                physics.named.data.qpos[f'rope/rJ1_{i + 1}'] = 0

    def current_action_vec(self, physics):
        left_tool_pos = physics.named.data.xpos['val/left_tool']
        right_tool_pos = physics.named.data.xpos['val/right_tool']
        right_tool_quat = physics.named.data.xquat['val/right_tool']
        left_tool_quat = physics.named.data.xquat['val/left_tool']
        return np.concatenate((left_tool_pos, left_tool_quat, right_tool_pos, right_tool_quat))

    def solve_ik(self, physics, target_pos, target_quat, site_name):
        # store the initial qpos to restore later
        initial_qpos = physics.bind(self.actuated_joints).qpos.copy()
        result = inverse_kinematics.qpos_from_site_pose(
            physics=physics,
            site_name=site_name,
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=self.actuated_joint_names,
            rot_weight=2,  # more rotation weight than the default
            # max_steps=10000,
            inplace=True,
        )
        qdes = physics.named.data.qpos[self.actuated_joint_names]
        # reset the arm joints to their original positions, because the above functions actually modify physics state
        physics.bind(self.actuated_joints).qpos = initial_qpos
        return result.success, qdes

    def release_rope(self, physics):
        physics.model.eq_active[:] = np.zeros(1)

    def grasp_rope(self, physics):
        physics.model.eq_active[:] = np.ones(1)

    def follow_trajectory(self, env, trajectory: JointTrajectory):
        traj_goal = make_follow_joint_trajectory_goal(trajectory)

        # Interpolate the trajectory to a fine resolution
        # if you set max_step_size to be large and position tolerance to be small, then things will be jerky
        if len(trajectory.points) == 0:
            rospy.loginfo("Ignoring empty trajectory")
            return True

        # construct a list of the tolerances in order of the joint names
        trajectory_joint_names = trajectory.joint_names
        tolerance = get_ordered_tolerance_list(trajectory_joint_names, traj_goal.path_tolerance)
        goal_tolerance = get_ordered_tolerance_list(trajectory_joint_names, traj_goal.goal_tolerance, is_goal=True)
        interpolated_points = interpolate_joint_trajectory_points(trajectory.points, max_step_size=0.01)

        if len(interpolated_points) == 0:
            rospy.loginfo("Trajectory was empty after interpolation")
            return True

        trajectory_point_idx = 0
        t0 = rospy.Time.now()
        while True:
            # tiny sleep lets the listeners process messages better, results in smoother following
            rospy.sleep(1e-3)
            dt = rospy.Time.now() - t0

            # get feedback
            new_waypoint = False
            obs = env._observation_updater.get_observation()
            actual_joint_positions = []
            for n in trajectory_joint_names:
                i = self.actuated_joint_names.index(f'val/{n}')
                actual_joint_positions.append(obs['joint_positions'][0, i])

            actual_point = JointTrajectoryPoint(positions=actual_joint_positions, time_from_start=dt)
            while trajectory_point_idx < len(interpolated_points) - 1 and is_waypoint_reached(actual_point, interpolated_points[trajectory_point_idx], tolerance):
                trajectory_point_idx += 1
                new_waypoint = True

            desired_point = interpolated_points[trajectory_point_idx]

            if trajectory_point_idx >= len(interpolated_points) - 1 and \
                    is_waypoint_reached(actual_point, desired_point, goal_tolerance):
                return True

            if new_waypoint:
                action_vec = self.action_vec_from_positions_and_names(desired_point.positions, trajectory_joint_names)
                for _ in range(100):
                    time_step = env.step(action_vec)
                    obs = time_step.observation
                    obs['joint_positions']

            # let the caller stop
            error = waypoint_error(actual_point, desired_point)
            print(1, f"{error} {desired_point.time_from_start.to_sec()} {dt.to_sec()}")
            # if desired_point.time_from_start.to_sec() > 0 and dt > desired_point.time_from_start * 5.0:
            #     if trajectory_point_idx == len(interpolated_points) - 1:
            #         stop_msg = f"timeout. expected t={desired_point.time_from_start.to_sec()} but t={dt.to_sec()}." \
            #                    + f" error to waypoint is {error}, goal tolerance is {goal_tolerance}"
            #     else:
            #         stop_msg = f"timeout. expected t={desired_point.time_from_start.to_sec()} but t={dt.to_sec()}." \
            #                    + f" error to waypoint is {error}, tolerance is {tolerance}"
            #
            #     # command the current configuration
            #     action_vec = self.action_vec_from_positions_and_names(actual_point.positions, trajectory_joint_names)
            #     print(dt.to_sec())
            #     env.step(action_vec)
            #     rospy.loginfo("Preempt requested, aborting.")
            #     rospy.logwarn(f"Stopped with message: {stop_msg}")
            #     return True

    def action_vec_from_positions_and_names(self, positions, trajectory_joint_names):
        action_vec = np.zeros(len(self.actuated_joint_names))
        for j, n in enumerate(trajectory_joint_names):
            i = self.actuated_joint_names.index(f'val/{n}')
            action_vec[i] = positions[j]
        return action_vec


@ros_init.with_ros("val_rope_task")
def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=250)
    task = ValRopeManipulation({
        'max_step_size':       0.001,
        'static_env_filename': 'car1.xml',
    })
    env = composer.Environment(task, random_state=0, time_limit=9999)
    viz = MujocoVisualizer()
    # from dm_control import viewer
    # viewer.launch(env)

    env.reset()

    # create a mujoco arm_robots hdt_michigan object
    # this can call env.step() in send_joint_command()
    # if all we want is planning, we just need to create a move group and call plan
    # from arm_robots.hdt_michigan import Val
    # val = Val()
    # val.set_execute(False)
    # start_state = RobotState()
    # start_state.joint_state.name = val.get_joint_names(group_name='whole_body')
    # start_state.joint_state.position = [0] * 20
    # plan = val.plan_to_joint_config(group_name='both_arms',
    #                                 joint_config=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                 start_state=start_state)
    # task.follow_trajectory(env, plan.planning_result.plan.joint_trajectory)

    # move to grasp
    _, qdes = task.solve_ik(env.physics,
                            target_pos=[0, task.rope.length_m / 2, 0.05],
                            target_quat=quaternion_from_euler(0, -np.pi, 0),
                            site_name='val/left_tool')
    for i in range(1000):
        env.step(qdes)

    # grasp!
    task.grasp_rope(env.physics)
    while True:
        env.step([0] * 20)

    # # lift up
    # _, qdes = task.solve_ik(target_pos=[0, 0, 0.5],
    #                         target_quat=quaternion_from_euler(0, -np.pi - 0.4, 0),
    #                         site_name='val/left_tool')
    # my_step(viz, env, [0] * 20, 20)

    # release
    task.release_rope(env.physics)
    for i in range(100):
        env.step([0] * 20)


if __name__ == "__main__":
    main()
