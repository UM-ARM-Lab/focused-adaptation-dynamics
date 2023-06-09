import os
import time
from typing import Dict, Optional, List

import numpy as np
import rospy
import torch
import transformations
from dm_envs.softgym_services import SoftGymServices
from link_bot_data.dataset_utils import add_predicted
from link_bot_pycommon.experiment_scenario import MockRobot
from link_bot_pycommon.get_occupancy import get_environment_for_extents_3d
from link_bot_pycommon.grid_utils_np import extent_res_to_origin_point
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_mp4
from visualization_msgs.msg import MarkerArray, Marker


class WaterSimScenario(ScenarioWithVisualization):
    """
    Representation:
    state: target/control pos, target/control volume
    goal: target_volume, tolerance
    action: goal x,z of control. theta, which can be 0.
    """

    def __init__(self, params: Optional[dict] = None):
        ScenarioWithVisualization.__init__(self, params)
        self.max_action_attempts = 300
        self._pos_tol = 0.002
        self._angle_tol = 0.05
        self.robot_reset_rng = np.random.RandomState(0)
        self.control_volumes = []
        self.target_volumes = []
        self.data_collect_id = 0
        # Hack for when you really don't want to run flex for environment reasons
        if "NO_FLEX" in os.environ:
            self.params["run_flex"] = False
        if self.params.get("run_flex", False):
            self._make_softgym_env()
        self.service_provider = SoftGymServices()
        if self.params.get("run_flex", False):
            self.service_provider.set_scene(self._scene)
        self.robot = MockRobot()
        self.robot.disconnect = lambda: self._scene.close()
        self.robot.jacobian_follower = None

    def _make_softgym_env(self):
        softgym_env_name = self.params.get("softgym_env_name", "PourWaterPlant")
        env_kwargs = env_arg_dict[softgym_env_name]

        default_config = {"save_frames": False, 'img_size': 10}
        self._save_cfg = self.params.get("save_cfg", default_config)
        # Generate and save the initial states for running this environment for the first time
        env_kwargs['use_cached_states'] = False
        env_kwargs['save_cached_states'] = False
        env_kwargs['num_variations'] = 1
        env_kwargs['render'] = False
        env_kwargs["action_repeat"] = 2
        env_kwargs['headless'] = not self.params.get('gui', False)

        if not env_kwargs['use_cached_states']:
            print('Waiting to generate environment variations. May take 1 minute for each variation...')
        self._scene = normalize(SOFTGYM_ENVS[softgym_env_name](**env_kwargs))
        self._save_frames = self._save_cfg["save_frames"]
        geom_params = self._scene.glass_params
        self.poured_dims = [geom_params["poured_glass_dis_x"], geom_params["poured_height"], geom_params["poured_glass_dis_z"]]
        self.pourer_dims = [geom_params["glass_dis_x"], geom_params["height"], geom_params["glass_dis_z"]]
        if self._save_frames:
            self.frames = []

    def needs_reset(self, state: Dict, params: Dict):
        if state["control_volume"].item() < 0.5:
            return True
        total_water_in_containers = state["control_volume"].item() + state["target_volume"].item()
        if total_water_in_containers < 0.5:
            return True
        return False

    def classifier_distance_torch(self, s1, s2):
        """ this is not the distance metric used in planning """
        container_dist = torch.linalg.norm(s1["controlled_container_pos"] - s2["controlled_container_pos"], axis=-1)
        target_volume_dist = torch.abs(s1["target_volume"] - s2["target_volume"])
        control_volume_dist = torch.abs(s1["control_volume"] - s2["control_volume"])
        target_volume_dist = target_volume_dist.squeeze(-1)
        control_volume_dist = control_volume_dist.squeeze(-1)

        return container_dist + 0.5 * target_volume_dist + 0.5 * control_volume_dist

    def classifier_distance(self, s1, s2):
        """ this is not the distance metric used in planning """
        container_dist = np.linalg.norm(s1["controlled_container_pos"] - s2["controlled_container_pos"], axis=-1)
        target_volume_dist = np.abs(s1["target_volume"] - s2["target_volume"])
        control_volume_dist = np.abs(s1["control_volume"] - s2["control_volume"])
        target_volume_dist = target_volume_dist.flatten()
        control_volume_dist = control_volume_dist.flatten()
        if not len(container_dist.shape):
            target_volume_dist = target_volume_dist.item()
            control_volume_dist = control_volume_dist.item()

        return container_dist + 0.5 * target_volume_dist + 0.5 * control_volume_dist

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

    def get_environment(self, params: Dict, **kwargs):
        res = params["res"]
        extent_key = "scenario_extent" if "scenario_extent" in params else "extent"
        voxel_grid_env = get_environment_for_extents_3d(extent=params[extent_key],
                                                        res=res,
                                                        frame='world',
                                                        service_provider=self.service_provider,
                                                        excluded_models=[])

        env = {}
        env.update({k: np.array(v).astype(np.float32) for k, v in voxel_grid_env.items()})
        env["origin_point"] = extent_res_to_origin_point(extent=params[extent_key], res=res)
        env["res"] = np.array(res)
        return env

    def on_before_get_state_or_execute_action(self):
        self.on_before_action()

    def on_before_action(self):
        pass

    def on_before_data_collection(self, params: Dict):
        self.reset()

    def interpolate(self, start_state, end_state, step_size=0.08):
        controlled_container_start = np.array(start_state['controlled_container_pos'])
        controlled_container_angle_start = start_state['controlled_container_angle'].item()
        controlled_container_angle_end = end_state['controlled_container_angle'].item()
        controlled_container_end = np.array(end_state['controlled_container_pos'])
        controlled_container_delta = controlled_container_end - controlled_container_start
        pos_steps = np.round(np.linalg.norm(controlled_container_delta) / step_size).astype(np.int64)
        angle_steps = np.round(
            np.abs(controlled_container_angle_end - controlled_container_angle_start) / step_size).astype(np.int64)
        steps = max(pos_steps, angle_steps)

        interpolated_actions = []
        angle_traj = np.linspace(controlled_container_angle_start, controlled_container_angle_end, steps)
        for idx, t in enumerate(np.linspace(step_size, 1, steps)):
            controlled_container_t = controlled_container_start + controlled_container_delta * t
            controlled_container_target_angle_interpolated = np.array([angle_traj[idx]])
            controlled_container_target_angle_interpolated = _match_2d_1d_tensor_shapes(controlled_container_t,
                                                                                        controlled_container_target_angle_interpolated)
            action = {
                'controlled_container_target_pos': controlled_container_t,
                'controlled_container_target_angle': controlled_container_target_angle_interpolated
            }
            interpolated_actions.append(action)
        if len(interpolated_actions) == 0:  # really nothing to interpolate, just give a "null action"
            controlled_container_angle_end_matched = _match_2d_1d_tensor_shapes(controlled_container_end, np.array(
                [controlled_container_angle_end]))
            null_action = {
                'controlled_container_target_pos': controlled_container_end,
                'controlled_container_target_angle': controlled_container_angle_end_matched,
            }
            return [null_action]
        return interpolated_actions

    def _on_execution_complete(self, fn, reached_goal=False, idx=0):
        if self._save_frames:
            save_name = f"{fn}_{reached_goal}_{idx}.mp4"
            save_numpy_as_mp4(np.array(self.frames), save_name)
            print("Saved to", save_name)
            self.frames = []

    def on_after_data_collection(self, params: Dict):
        self.data_collect_id += 1
        if self._save_frames:
            save_name = f"mp4s/data_collect_{self.data_collect_id}.mp4"
            save_numpy_as_mp4(np.array(self.frames), save_name)
            print("Saved to", save_name)
            self.frames = []

    def execute_compound_action(self, environment, state: Dict, action: Dict, **kwargs):
        local_action = self.put_action_local_frame(state, action)
        delta_size = np.linalg.norm(local_action['delta_pos'])
        if delta_size < self.params["max_move_dist"]:
            self.execute_action(environment, state, action, **kwargs)
        else:
            # Not 100% accurate split but less prone to worse error
            min_num_actions = int(np.ceil(delta_size / self.params["max_move_dist"]))
            # each row is an action
            smaller_actions_np = np.linspace(state["controlled_container_pos"],
                                             action["controlled_container_target_pos"], min_num_actions)
            for smaller_action_np in smaller_actions_np:
                smaller_action = {
                    "controlled_container_target_pos": smaller_action_np,
                    "controlled_container_target_angle": action["controlled_container_target_angle"],
                }
                current_state = self.get_state()
                self.execute_action(environment, current_state, smaller_action)

    def execute_action(self, environment, state: Dict, action: Dict, **kwargs):
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
            if np.linalg.norm(curr_pos - goal_pos) < self._pos_tol and np.abs(
                    goal_angle - curr_angle) < self._angle_tol:
                if curr_state["control_volume"] > 0.9 or curr_state["target_volume"] > 0.99:
                    break
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
            action_types = ["free_space", "over_target", "tilt"]
            if self.is_pour_valid_for_state(state):
                action_type_probs = [0.2, 0.1, 0.7]
            elif state["controlled_container_pos"][1] < 0.02:  # on ground only one thing works
                action_type_probs = [0.7, 0.3, 0.0]
            else:
                action_type_probs = [0.3, 0.7, 0.0]

            action_type = np.random.choice(action_types, p=action_type_probs)
            if action_type == "tilt":
                random_angle = np.array([action_rng.uniform(low=self.params["action_range"]["angle"][0],
                                                            high=self.params["action_range"]["angle"][1])],
                                        dtype=np.float32)
                action = {"controlled_container_target_pos": current_controlled_container_pos,
                          "controlled_container_target_angle": random_angle}
            elif action_type == "over_target":
                noise = action_rng.uniform(low=-0.1, high=0.1, size=(2,))
                des_height = self._scene.glass_params["poured_height"] + self._scene.glass_params["height"] + 0.1
                des_x = self._scene.glass_params["poured_glass_x_center"] - self._scene.glass_params[
                    "poured_glass_dis_x"] / 3
                over_box_pose_with_noise = np.array([des_x, des_height], dtype=np.float32) + noise
                action = {"controlled_container_target_pos": over_box_pose_with_noise,
                          "controlled_container_target_angle": np.array([0], dtype=np.float32)}
            else:
                random_x = action_rng.uniform(low=self.params["action_range"]["x"][0],
                                              high=self.params["action_range"]["x"][1])
                random_y = action_rng.uniform(low=self.params["action_range"]["y"][0],
                                              high=self.params["action_range"]["y"][1])
                action = {"controlled_container_target_pos": np.array([random_x, random_y], dtype=np.float32),
                          "controlled_container_target_angle": np.array([0], dtype=np.float32)}
            if validate and self.is_action_valid(environment, state, action, action_params):
                return action, False
        return None, True

    def is_action_valid(self, environment: Dict, state: Dict, action: Dict, action_params: Dict):
        curr_pos = state["controlled_container_pos"]
        target_pos = action["controlled_container_target_pos"]
        if np.linalg.norm(curr_pos - target_pos) > self.params["max_move_dist"]:
            return False
        is_pour = action["controlled_container_target_angle"] >= 0.001
        if is_pour:
            return self.is_pour_valid_for_state(state)
        return True

    def is_pour_valid_for_state(self, state: Dict):
        min_height_for_pour = self._scene.glass_params["poured_height"] + 0.02
        target_x = state["target_container_pos"][0] - self._scene.glass_params["poured_glass_dis_x"] / 2.
        curr_pos = state["controlled_container_pos"]
        curr_height = curr_pos[1]
        if curr_height < min_height_for_pour:
            return False
        x_dist_between_center_and_edge = np.abs(curr_pos[0] - target_x)
        if x_dist_between_center_and_edge > self.params["max_dist_for_pour"]:
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
        pos_xy = state["controlled_container_pos"].unsqueeze(-1)
        local_center = torch.cat((pos_xy[:, 0], pos_xy[:, 1], torch.zeros_like(pos_xy[:, 0])), dim=1)
        if len(local_center.shape) == 0:
            return local_center.reshape(1, -1)
        return local_center

    def reset(self, env_rng=None):
        self._scene.reset()
        null_action = np.zeros(3, )
        self._saved_data = self._scene.step(null_action, record_continuous_video=self._save_frames,
                                            img_size=self._save_cfg["img_size"])
        if env_rng is None:
            env_rng = np.random
        if self.params.get("randomize_start", False):
            state = self.get_state()
            for j in range(40):
                random_x = env_rng.uniform(low=self.params["action_range"]["x"][0],
                                             high=self.params["action_range"]["x"][1])
                random_y = env_rng.uniform(low=self.params["action_range"]["y"][0],
                                             high=self.params["action_range"]["y"][1])
                action = {'controlled_container_target_pos': np.array([random_x, random_y]),
                          'controlled_container_target_angle': env_rng.uniform(low=-0.05, high=0.1, size=(1,))}

                if self.is_moveit_robot_in_collision_range(None, state, action, 0.03):
                    print("In collision trying again")
                else:
                    self.execute_compound_action(None, state, action)
                    break
                if j == 40:
                    print("Could not sample valid start state")
                    self.reset()
        init_state = self.get_state()
        if not (init_state["target_volume"] == 0 and init_state["control_volume"] == 1.0):
            print("Water spilled: resetting again")
            self.reset()
         
        curr_pos = init_state["controlled_container_pos"]
        curr_angle = init_state["controlled_container_angle"]
        pose = np.hstack([curr_pos, curr_angle]).flatten()
        if self._scene._wrapped_env.predict_collide_with_plant(pose):
            print("Got stuck: resetting again")
            self.reset()

    def randomize_environment(self, env_rng, params: Dict):
        self.reset(env_rng)

    def reset_to_start(self, planner_params, _):
        self.reset()

    @staticmethod
    def put_state_robot_frame(state: Dict):
        # Assumes everything is in robot frame already
        return state

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_pos = action["controlled_container_target_pos"]
        target_angle = action["controlled_container_target_angle"]

        current_pos = state["controlled_container_pos"]
        current_angle = state["controlled_container_angle"]
        target_angle = _match_2d_1d_tensor_shapes(current_angle, target_angle)
        assert target_angle.shape == current_angle.shape

        delta_pos = target_pos - current_pos
        delta_angle = target_angle - current_angle

        delta_angle = _match_2d_1d_tensor_shapes(delta_pos, delta_angle)
        assert len(delta_pos.shape) == len(delta_angle.shape)
        if not isinstance(target_pos, np.ndarray):  # must be a tensor
            delta_pos = delta_pos.float()
            delta_angle = delta_angle.float()
        return {
            'delta_pos': delta_pos,
            'delta_angle': delta_angle
        }

    @staticmethod
    def apply_local_action_at_state(state, local_action):
        delta_pos = state['delta_pos']
        delta_angle = state['delta_angle']
        curr_angle = _fix_extremes_1d_data(state["current_controlled_container_angle"])

        local_action = {"controlled_container_target_pos": state["current_controlled_container_pos"] + delta_pos,
                        "controlled_container_target_angle": (
                                curr_angle + delta_angle)}
        local_action["controlled_container_target_angle"] = _match_2d_1d_tensor_shapes(
            local_action["controlled_container_target_pos"], local_action["controlled_container_target_angle"])
        return local_action

    @staticmethod
    def put_state_local_frame_torch(state: Dict):
        target_pos = state["target_container_pos"]
        current_pos = state["controlled_container_pos"]
        current_angle = state["controlled_container_angle"]
        delta_pos = target_pos - current_pos

        assert len(delta_pos.shape) == len(current_angle.shape)
        local_dict = {
            'controlled_container_pos_local': delta_pos.float(),
            'controlled_container_angle_local': (current_angle).float(),
        }
        local_dict["target_volume"] = state["target_volume"].float()
        local_dict["control_volume"] = state["control_volume"].float()
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

    def is_moveit_robot_in_collision(self, environment: Dict, state: Dict, action: Dict):
        target_pos = action["controlled_container_target_pos"]
        target_angle = action["controlled_container_target_angle"]
        target_pose = np.hstack([target_pos, target_angle]).flatten()
        return self._scene._wrapped_env.predict_collide_with_plant(target_pose)

    def is_moveit_robot_in_collision_range(self, environment: Dict, state: Dict, action: Dict, range : float):
        #change x , and y
        original_target_pos = action["controlled_container_target_pos"]
        range_action = {'controlled_container_target_angle': action["controlled_container_target_angle"]}
        deltas = np.array([[0,range],[0, -range],[range, 0],[-range,0]])
        for delta in deltas:
            range_action["controlled_container_target_pos"] = original_target_pos + delta
            if self.is_moveit_robot_in_collision(environment, state, range_action):
                return True
        return False

    def moveit_robot_reached(self, state: Dict, action: Dict, next_state: Dict):
        # somewhat of a lie because no moveit
        return True

    def can_interpolate(self, state: Dict, next_state: Dict):
        # Checks if it can reach the next state using our controllers, which don't do angle + pos movement. Since interpolate doesn't use the action we only do state here
        max_theta_for_move = 0.1
        max_move_for_pour = 0.07  # TODO make these configs?
        curr_pos = state["controlled_container_pos"]
        curr_angle = state["controlled_container_angle"].item()
        next_pos = next_state["controlled_container_pos"]
        next_angle = next_state["controlled_container_angle"].item()

        # first check if action intends it to be a pour
        if abs(next_angle - curr_angle) > max_theta_for_move:
            if np.linalg.norm(curr_pos - next_pos) > max_move_for_pour:
                return False
        return True

    def get_state(self):
        state_vector = self._saved_data[0][0]
        state = {"controlled_container_pos": state_vector[:2],
                 "controlled_container_angle": np.array([state_vector[2]], dtype=np.float32),
                 "target_container_pos": np.array([state_vector[6] - state_vector[0], 0]),
                 "control_volume": np.array([state_vector[-1]], dtype=np.float32),
                 "target_volume": np.array([state_vector[-2]], dtype=np.float32)}
        return state

    def make_box_marker(self, pose, dims, rgb, angle = 0, alpha=1):
        marker = Marker()
        marker.scale.x = dims[0]
        marker.scale.y = dims[1]
        marker.scale.z = dims[2]
        marker.action = Marker.ADD
        marker.type = Marker.CUBE
        marker.pose.position.x = pose[0]  # y is 0
        marker.pose.position.y = pose[1]
        quat_wxyz = transformations.quaternion_from_euler(0,0,-angle)
        marker.pose.orientation.w = quat_wxyz[0]
        marker.pose.orientation.x = quat_wxyz[1]
        marker.pose.orientation.y = quat_wxyz[2]
        marker.pose.orientation.z = quat_wxyz[3]
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = alpha
        return marker

    def make_volume_marker(self, pose, volume, label="volume", volume_status="default", alpha=1):
        marker = Marker()
        marker.action = Marker.ADD
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose.position.x = pose[0] + 0.03  # y is 0
        marker.pose.position.y = pose[1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = label
        marker.text = f"volume: {volume.round(2)}"
        marker.color.a = alpha
        if volume_status == "spilled":
            color =  [1,0,0]
        elif volume_status == "filled":
            color = [0,0.2,1]
            marker.lifetime.secs = 1
        else:
            color = [.6, .6, .6]
        marker.color.r = color[0]
        marker.color.g = color[2]
        marker.color.b = color[1]
        marker.scale.z = 0.2
        return marker

    def make_angle_marker(self, pose, angle):
        marker = Marker()
        marker.action = Marker.ADD
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose.position.x = pose[0] + 0.03  # y is 0
        marker.pose.position.y = pose[1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        #marker.ns = "angle"
        marker.text = f"angle: {angle.round(2)}"
        marker.color.a = 1
        marker.color.b = 0.2
        marker.color.r = 0.9
        marker.color.g = 0.9
        marker.scale.z = 0.1
        return marker

    def make_state_msg(self, state, target_pos=None, target_angle=None):
        msg = MarkerArray()
        if target_pos is None:
            if add_predicted("controlled_container_pos") in state:
                pourer_pos = state[add_predicted("controlled_container_pos")]
                pourer_angle = state[add_predicted("controlled_container_angle")]
                alpha = 0.5
            else:
                pourer_pos = state["controlled_container_pos"]
                pourer_angle = state["controlled_container_angle"]
                alpha = 1.0
        else:
            pourer_pos = target_pos
            pourer_angle = target_angle
            alpha=0.1 #very light for actions

        if "control_volume" in state:
            control_volume = state["control_volume"]
            target_volume = state["target_volume"]
        else:
            control_volume = state[add_predicted("control_volume")]
            target_volume = state[add_predicted("target_volume")]
        if control_volume + target_volume < 0.97:
            volume_status = "spilled"
        elif target_volume > 0.97:
            volume_status = "filled"
        else:
            volume_status = "default"
        pourer_dims = self.pourer_dims
        poured_dims = self.poured_dims
        pourer_marker = self.make_box_marker(pourer_pos, pourer_dims, angle=pourer_angle, rgb=np.array([1, 0, 0]), alpha=alpha)
        pourer_marker.id = 0
        msg.markers.append(pourer_marker)
        pourer_volume_marker = self.make_volume_marker(pourer_pos, control_volume, "control_volume", alpha=alpha, volume_status=volume_status)
        pourer_volume_marker.id = 2
        msg.markers.append(pourer_volume_marker)
        if add_predicted("target_container_pos") in state:
            poured_pos = state[add_predicted("target_container_pos")]
        else:
            poured_pos = state["target_container_pos"]
        poured_marker = self.make_box_marker(poured_pos, poured_dims, rgb=np.array([0, 1, 0]), alpha=alpha)
        poured_volume_marker = self.make_volume_marker(poured_pos, target_volume, "target_volume", alpha=alpha, volume_status=volume_status)
        poured_marker.id = 3
        poured_volume_marker.id = 4
        msg.markers.append(poured_marker)
        msg.markers.append(poured_volume_marker)
        return msg

    def plot_state_rviz(self, state: Dict, **kwargs):
        state_marker_msg = self.make_state_msg(state)
        self.state_viz_pub.publish(state_marker_msg)

    def plot_goal_rviz(self, goal, threshold, **kwargs):
        pass  # TODO plot something to show goal range. Low priority since this doesn't change much

    def plot_action_rviz(self, state: Dict, action: Dict, label: str = 'action', **kwargs):
        target_pos = action["controlled_container_target_pos"]
        target_angle = action["controlled_container_target_angle"]
        state_marker_msg = self.make_state_msg(state, target_pos=target_pos, target_angle=target_angle)
        self.action_viz_pub.publish(state_marker_msg)

    def distance_to_goal(self, state: Dict, goal: Dict, use_torch=False):
        goal_target_volume_range = goal["goal_target_volume_range"]
        curr_target_volume = abs(state["target_volume"].item())
        curr_control_volume = abs(state["control_volume"].item())
        total_volume = curr_target_volume + curr_control_volume
        if curr_target_volume > goal_target_volume_range[0] and curr_target_volume < goal_target_volume_range[1]:
            desired_volume_dist = 0
        else:
            too_low_amount = abs(curr_target_volume - goal_target_volume_range[0])
            too_high_amount = abs(curr_target_volume - goal_target_volume_range[1])
            desired_volume_dist = min(too_low_amount, too_high_amount)
        spill_penalty = 3
        amount_spilled = np.abs(1 - total_volume)
        if amount_spilled < 0.02 or total_volume > 1:
            # negligible
            amount_spilled = 0
        desired_spill_dist = min(0.25, spill_penalty * amount_spilled)
        return desired_spill_dist + desired_volume_dist

    def distance_to_goal_pos(self, state: Dict, goal: Dict, use_torch=False):
        goal_pos = goal["controlled_container_pos"]
        curr_pos = state["controlled_container_pos"]
        distance = ((goal_pos - curr_pos) ** 2).sum() ** 0.5
        return distance

    def simple_name(self):
        return "watering"

    def __repr__(self):
        return self.simple_name()


# Specific to water scenario. A lot of functions seem to assume 2D data and then it gets converted
def _fix_extremes_1d_data(data):
    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            data = data.reshape(data.shape + (1,))
            return data
    else:
        if len(data.shape) == 1:
            data = data.unsqueeze(-1)
            return data
        if len(data.shape) == 3:
            data = data.squeeze(-1)
            return data
    return data


def _squeeze_if_3d(data):
    if len(data.shape) == 3:
        data = data.squeeze(-1)
    return data


def _match_2d_1d_tensor_shapes(tensor_to_match, tensor_needing_matching):
    # Patch fix that only really seems to come up here w/ 1D data
    desired_shape_size = len(tensor_to_match.shape)
    current_shape_size = len(tensor_needing_matching.shape)
    assert abs(desired_shape_size - current_shape_size) <= 1
    if current_shape_size > desired_shape_size:
        matched_tensor = tensor_needing_matching.squeeze(-1)
    elif current_shape_size < desired_shape_size:
        matched_tensor = tensor_needing_matching.unsqueeze(-1)
    else:
        matched_tensor = tensor_needing_matching

    matched_tensor_shape_size = len(matched_tensor.shape)
    assert matched_tensor_shape_size == desired_shape_size
    return matched_tensor
