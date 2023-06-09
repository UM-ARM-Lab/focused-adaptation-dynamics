import pathlib
import re
from typing import Optional, List

import rospkg

import ros_numpy
import rosbag
import roslaunch
import rospy
from arm_gazebo_msgs.srv import SetLinkStates, SetLinkStatesRequest, GetWorldInitialSDFResponse, GetWorldInitialSDF
from gazebo_msgs import msg as gz_msg
from gazebo_msgs.srv import SetPhysicsPropertiesRequest, GetPhysicsPropertiesRequest

from link_bot_gazebo import gazebo_utils
from link_bot_pycommon.base_services import BaseServices
from std_srvs.srv import EmptyRequest, Empty


class GazeboServices(BaseServices):

    def __init__(self):
        super().__init__()
        self.max_step_size = None
        self.gazebo_process = None

        # Yes, absolute paths here are what I want. I don't want these namespaced by the robot
        self.set_link_states = self.add_required_service('arm_gazebo/set_link_states', SetLinkStates)
        self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty, persistent=True)
        self.play_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty, persistent=True)
        self.get_world_initial_sdf_srv = rospy.ServiceProxy('arm_gazebo/world_initial_sdf', GetWorldInitialSDF)

    def restore_from_bag(self, bagfile_name: pathlib.Path, excluded_models: Optional[List[str]] = None):
        with rosbag.Bag(bagfile_name) as bag:
            saved_links_states: gz_msg.LinkStates = next(iter(bag.read_messages()))[1]
        self.restore_from_link_states_msg(saved_links_states, excluded_models)

    def restore_from_link_states_msg(self, saved_links_states, excluded_models=[]):
        set_states_req = SetLinkStatesRequest()
        for name, pose, twist in zip(saved_links_states.name, saved_links_states.pose, saved_links_states.twist):
            model_name, link_name = name.split('::')
            if excluded_models is not None and model_name not in excluded_models:
                link_state = gz_msg.LinkState()
                # name here is model_name::link_name, what gazebo calls "scoped"
                link_state.link_name = name
                link_state.pose = pose
                link_state.twist = twist
                set_states_req.link_states.append(link_state)
        res = self.set_link_states(set_states_req)
        if not res.success:
            raise RuntimeError(f"Failed to restore state! {res}")

    def launch(self, params, **kwargs):
        gui = kwargs.get("gui", True)
        world = kwargs.get("world", None)
        launch_file_name = params['launch']

        rospack = rospkg.RosPack()
        full_world_file = pathlib.Path(rospack.get_path('link_bot_gazebo')) / 'worlds' / world
        if not full_world_file.exists():
            rospy.logerr(f"No world file {full_world_file.as_posix()}")
            return (success := False)

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        roslaunch_args = ['link_bot_gazebo', launch_file_name, f"gui:={str(gui).lower()}", "--no-summary"]
        if world:
            roslaunch_args.append(f"world:={world}")

        roslaunch_file = roslaunch.rlutil.resolve_launch_arguments(roslaunch_args)[0]
        roslaunch_args = roslaunch_args[2:]

        launch_info = [(roslaunch_file, roslaunch_args)]

        self.gazebo_process = roslaunch.parent.ROSLaunchParent(uuid, launch_info)
        self.gazebo_process.start()

        # wait until services are available before returning
        self.wait_for_services()

        self.play()
        return (success := True)

    def kill(self):
        self.gazebo_process.shutdown()

    def setup_env(self, verbose: int = 0, real_time_rate: float = 0, max_step_size: float = 0.01, play: bool = True):
        gazebo_utils.resume()

        # set up physics
        get_physics_msg = GetPhysicsPropertiesRequest()
        current_physics = self.get_physics.call(get_physics_msg)
        set_physics_msg = SetPhysicsPropertiesRequest()
        set_physics_msg.gravity = current_physics.gravity
        set_physics_msg.ode_config = current_physics.ode_config
        set_physics_msg.max_update_rate = real_time_rate / max_step_size
        if max_step_size is None:
            max_step_size = current_physics.time_step
        self.max_step_size = max_step_size
        set_physics_msg.time_step = max_step_size
        self.set_physics.call(set_physics_msg)

        if play:
            self.play()

    def play(self):
        gazebo_utils.resume()
        try:
            self.play_srv(EmptyRequest())
        except rospy.ServiceException:
            pass

    def pause(self):
        gazebo_utils.resume()
        try:
            self.pause_srv(EmptyRequest())
        except rospy.ServiceException:
            pass

    def get_world_initial_sdf(self):
        gazebo_utils.resume()
        res: GetWorldInitialSDFResponse = self.get_world_initial_sdf_srv()
        sdf = res.world_initial_sdf
        sdf = sdf.replace("\n", "")
        sdf = re.sub("> +", ">", sdf)
        return sdf


def gz_scope(*args):
    return "::".join(args)


def restore_gazebo(gz, link_states, s):
    restore_action = {
        'left_gripper_position':  ros_numpy.numpify(
            link_states.pose[link_states.name.index('rope_3d::left_gripper')].position),
        'right_gripper_position': ros_numpy.numpify(
            link_states.pose[link_states.name.index('rope_3d::right_gripper')].position),
    }
    s.execute_action(None, None, restore_action, wait=False)
    gz.pause()
    gz.restore_from_link_states_msg(link_states, excluded_models=['kinect2', 'collision_sphere', 'my_ground_plane'])
    gz.play()
