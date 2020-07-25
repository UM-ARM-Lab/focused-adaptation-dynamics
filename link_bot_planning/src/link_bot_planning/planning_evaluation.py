import gzip
import json
import pathlib
import uuid
from time import sleep, time
from typing import Optional, Dict, List

import rospy
import rosbag
import numpy as np
from dataclasses_json import dataclass_json
from dataclasses_json import DataClassJsonMixin
from control_msgs.msg import FollowJointTrajectoryActionGoal
import ompl.util as ou
import tensorflow as tf
from colorama import Fore
from ompl import base as ob
from time import perf_counter
from victor import victor_services
from moonshine.moonshine_utils import listify
from link_bot_pycommon.base_services import BaseServices
from link_bot_gazebo_python import gazebo_services
from link_bot_planning.ompl_viz import planner_data_to_json
from link_bot_planning.my_planner import MyPlanner, MyPlannerStatus, PlanningResult
from link_bot_planning.get_planner import get_planner
from link_bot_pycommon.serialization import dummy_proof_write
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_planning import plan_and_execute


class EvalPlannerConfigs(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 service_provider: BaseServices,
                 planner_config_name: str,
                 n_trials: int,
                 verbose: int,
                 planner_params: Dict,
                 comparison_item_idx: int,
                 seed: int,
                 goal,
                 outdir: pathlib.Path,
                 record: Optional[bool] = False,
                 no_execution: Optional[bool] = False,
                 ):
        super().__init__(planner,
                         n_trials=n_trials,
                         verbose=verbose,
                         planner_params=planner_params,
                         service_provider=service_provider,
                         no_execution=no_execution,
                         seed=seed)
        self.record = record
        self.planner_config_name = planner_config_name
        self.outdir = outdir
        self.seed = seed

        self.subfolder = "{}_{}".format(
            self.planner_config_name, comparison_item_idx)
        self.root = self.outdir / self.subfolder
        self.root.mkdir(parents=True)
        print(Fore.CYAN + str(self.root) + Fore.RESET)
        self.goal = goal

        metadata = {
            "n_trials": self.n_trials,
            "planner_params": self.planner_params,
            "scenario": self.planner.scenario.simple_name(),
            "seed": self.seed,
            "horizon": self.planner.classifier_model.horizon,
        }
        with (self.root / 'metadata.json').open("w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        self.joint_goal_sub = rospy.Subscriber("/both_arms_controller/follow_joint_trajectory/goal",
                                               FollowJointTrajectoryActionGoal,
                                               self.follow_joint_trajectory_goal_callback,
                                               queue_size=10)
        self.bag = None

    def randomize_environment(self):
        super().randomize_environment()

    def on_start_trial(self):
        if self.record:
            filename = self.root.absolute() / 'plan-{}.avi'.format(self.trial_idx)
            # self.service_provider.start_record_trial(str(filename))
            bagname = self.root.absolute() / f"follow_joint_trajectory_goal_{self.trial_idx}.bag"
            print(Fore.YELLOW + str(bagname) + Fore.RESET)
            self.bag = rosbag.Bag(bagname, 'w')

    def follow_joint_trajectory_goal_callback(self, goal_msg):
        self.bag.write('/both_arms_controller/follow_joint_trajectory/goal', goal_msg)
        self.bag.flush()

    def get_goal(self, environment: Dict):
        if self.goal is not None:
            if self.verbose >= 1:
                print("Using Goal {}".format(self.goal))
            return self.goal
        else:
            return super().get_goal(environment)

    def on_trial_complete(self, trial_data: Dict):
        extra_trial_data = {
            "planner_params": self.planner_params,
            "scenario": self.planner.scenario.simple_name(),
            "seed": self.seed,
            'current_time': int(time()),
            'uuid': uuid.uuid4(),
        }
        trial_data.update(extra_trial_data)
        data_filename = self.root / f'{self.trial_idx}_metrics.json.gz'
        dummy_proof_write(trial_data, data_filename)

        if self.record:
            # TODO: maybe make this happen async?
            # sleep(1)
            # self.service_provider.stop_record_trial()
            self.bag.close()


def evaluate_planning_method(args, comparison_idx, planner_params, p_params_name, common_output_directory):
    # start at the same seed every time to make the planning environments & plans the same (hopefully?)
    # setting OMPL random seed should have no effect, because I use numpy's random in my sampler?
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)  # not sure if this has any effect

    # override some arguments
    if args.timeout is not None:
        rospy.loginfo(f"Overriding with timeout {args.timeout}")
        planner_params["termination_criteria"]['timeout'] = args.timeout

    planner_config_name = p_params_name.stem

    # Start Services
    if args.env_type == 'victor':
        service_provider = victor_services.VictorServices()
    elif args.env_type == 'val':
        service_provider = victor_services.VictorServices()
    else:
        service_provider = gazebo_services.GazeboServices()

    # look up the planner params
    planner, _ = get_planner(planner_params=planner_params,
                             seed=args.seed,
                             verbose=args.verbose)

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size)

    print(Fore.GREEN + "Running {} Trials".format(args.n_trials) + Fore.RESET)

    runner = EvalPlannerConfigs(
        planner=planner,
        service_provider=service_provider,
        planner_config_name=planner_config_name,
        n_trials=args.n_trials,
        verbose=args.verbose,
        planner_params=planner_params,
        seed=args.seed,
        outdir=common_output_directory,
        comparison_item_idx=comparison_idx,
        goal=planner_params['fixed_goal'],
        record=args.record,
        no_execution=args.no_execution
    )
    runner.run()
