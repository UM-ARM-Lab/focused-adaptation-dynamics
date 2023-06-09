import pathlib
import re
from typing import Optional

import numpy as np

import rosbag
import rospy
from gazebo_msgs.msg import LinkStates
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_hjson
from moonshine.numpify import numpify
from sensor_msgs.msg import JointState


def make_scene_filename(root, idx):
    in_scene_file = root / f'scene_{idx:04d}.bag'
    return in_scene_file


def make_goal_filename(root, idx):
    in_goal_file = root / f'goal_{idx:04d}.hjson'
    return in_goal_file


def get_all_scene_indices(dirname: pathlib.Path):
    for s in get_all_scenes(dirname):
        yield s.idx


def get_all_scenes(dirname: pathlib.Path):
    scenes = []
    goal_filenames = sorted(list(dirname.glob("goal_*.hjson")))
    for goal_filename in goal_filenames:
        m = re.match(r'goal_(\d+).hjson', goal_filename.name)
        idx = int(m.group(1))
        scene = TestScene(dirname, idx)
        scenes.append(scene)
    return scenes


def save_goal(goal, root, idx):
    goal_filename = make_goal_filename(root, idx)
    my_hdump(goal, goal_filename.open("w"))


def load_goal(root, idx):
    goal_filename = make_goal_filename(root, idx)
    goal = load_hjson(goal_filename)
    return numpify(goal)


class TestScene:

    def __init__(self, root: pathlib.Path, idx: int):
        self.root = root
        self.idx = idx
        goal = load_goal(self.root, self.idx)

        scene_filename = make_scene_filename(self.root, self.idx)

        # read the data
        try:
            with rosbag.Bag(scene_filename) as bag:
                self.joint_state = next(iter(bag.read_messages(topics=['joint_state'])))[1]
                self.links_states = next(iter(bag.read_messages(topics=['links_states'])))[1]
        except FileNotFoundError:
            rospy.logwarn_once("No bag file of link states and joint states")
            pass

        self.goal = {k: np.array(v, dtype=np.float32) for k, v in goal.items()}

    def get_scene_filename(self):
        scene_filename = make_scene_filename(self.root, self.idx)
        return scene_filename

    def save(self, force: Optional[bool] = False):
        save_goal(self.goal, self.root, self.idx)
        scene_filename = make_scene_filename(self.root, self.idx)
        save_test_scene_given_name(joint_state=self.joint_state,
                                   links_states=self.links_states,
                                   bagfile_name=scene_filename,
                                   force=force)

    def delete(self):
        goal_filename = make_goal_filename(self.root, self.idx)
        scene_filename = make_scene_filename(self.root, self.idx)
        scene_filename.unlink()
        goal_filename.unlink()

    def change_index(self, new_idx: int, force: Optional[bool] = False):
        print(f'{self.idx}->{new_idx}')

        out_scene_file = make_scene_filename(self.root, new_idx)

        out_goal_file = make_goal_filename(self.root, new_idx)

        if not force:
            if out_scene_file.exists():
                raise RuntimeError(f"file {out_scene_file.as_posix()} exists!")
            if out_goal_file.exists():
                raise RuntimeError(f"file {out_goal_file.as_posix()} exists!")

        self.idx = new_idx
        self.save(force=force)

    def remove(self):
        goal_filename = make_goal_filename(self.root, self.idx)
        scene_filename = make_scene_filename(self.root, self.idx)
        goal_filename.unlink(missing_ok=False)
        scene_filename.unlink(missing_ok=False)
        print(f"Removed {goal_filename.as_posix()} and {scene_filename.as_posix()}")


def save_test_scene(joint_state: JointState,
                    links_states: LinkStates,
                    save_test_scenes_dir: pathlib.Path,
                    trial_idx: int,
                    force: bool):
    bagfile_name = make_scene_filename(save_test_scenes_dir, trial_idx)
    return save_test_scene_given_name(joint_state, links_states, bagfile_name, force)


def save_test_scene_given_name(joint_state: JointState,
                               links_states: LinkStates,
                               bagfile_name: pathlib.Path,
                               force: bool):
    if bagfile_name.exists() and not force:
        print(f"File {bagfile_name.as_posix()} already exists. Aborting")
        return None

    rospy.logdebug(f"Saving scene to {bagfile_name}")
    with rosbag.Bag(bagfile_name, 'w') as bag:
        bag.write('links_states', links_states)
        bag.write('joint_state', joint_state)

    return bagfile_name


def create_randomized_start_state(params, scenario, trial_idx):
    env_rng = np.random.RandomState(trial_idx)
    action_rng = np.random.RandomState(trial_idx)
    environment = scenario.get_environment(params)
    scenario.randomize_environment(env_rng, params)
    for i in range(10):
        state = scenario.get_state()
        action, invalid = scenario.sample_action(action_rng=action_rng,
                                                 environment=environment,
                                                 state=state,
                                                 action_params=params,
                                                 validate=True,
                                                 )
        scenario.execute_action(None, None, action)


def get_states_to_save():
    links_states: LinkStates = rospy.wait_for_message("gazebo/link_states", LinkStates)
    joint_state: JointState = rospy.wait_for_message("hdt_michigan/joint_states", JointState)
    return joint_state, links_states
