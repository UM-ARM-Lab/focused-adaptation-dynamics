#!/usr/bin/env python
import argparse
import logging
import pathlib
import re
from typing import List

import tensorflow as tf

import rospy
from arc_utilities import ros_init
from link_bot_classifiers.train_test_classifier import eval_generator
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import deserialize_scene_msg
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.marker_index_generator import marker_index_generator
from link_bot_pycommon.matplotlib_utils import adjust_lightness
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import load_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.indexing import index_time_with_metadata
from moonshine.moonshine_utils import remove_batch
from moonshine.numpify import numpify
from peter_msgs.msg import VizOptions
from peter_msgs.srv import GetVizOptionsResponse, GetVizOptions, GetVizOptionsRequest


def eval_classifier_no_batch(scenario: ScenarioWithVisualization,
                             checkpoint: pathlib.Path,
                             iteration_dataset_dir: pathlib.Path,
                             mode: str):
    data_for_classifier_on_dataset = eval_generator(scenario=scenario,
                                                    dataset_dirs=[iteration_dataset_dir],
                                                    checkpoint=checkpoint,
                                                    mode=mode,
                                                    balance=False,
                                                    verbose=-1,
                                                    batch_size=1,
                                                    use_gt_rope=True)
    data_no_batch = []
    for e, o in data_for_classifier_on_dataset:
        deserialize_scene_msg(e)
        data_no_batch.append((numpify(remove_batch(e)), numpify(remove_batch(o))))
    return data_no_batch


def get_data(scenario: ScenarioWithVisualization,
             ift_dir: pathlib.Path,
             dataset_iteration_idx: int,
             classifier_iteration_idx: int,
             mode: str):
    classifier_datasets_dir = ift_dir / 'classifier_datasets'
    classifiers_dir = ift_dir / 'training_logdir'
    log = load_hjson(ift_dir / 'logfile.hjson')

    iteration_dataset_dir = get_named_item_in_dir(classifier_datasets_dir, classifier_iteration_idx)

    if classifier_iteration_idx == 0:
        initial_classifier_dir = log['initial_classifier_checkpoint']
        pretraining_log = log.get("pretraining", None)
        if pretraining_log is None:
            pretrained_classifier_dir = initial_classifier_dir
        else:
            pretrained_classifier_dir = pretraining_log.get("new_latest_checkpoint_dir", None)
            if pretrained_classifier_dir is None:
                pretrained_classifier_dir = initial_classifier_dir

        checkpoint = pathlib.Path(pretrained_classifier_dir)
    else:
        checkpoint = get_named_item_in_dir(classifiers_dir, classifier_iteration_idx - 1)
        checkpoint = next(checkpoint.iterdir())

    checkpoint = checkpoint / log['checkpoint_suffix']
    return eval_classifier_no_batch(scenario, checkpoint, iteration_dataset_dir, mode)


class MyFilter:

    def __init__(self, filter_names: List[str]):
        self.filter_names = filter_names
        self.get_viz_options_srv = rospy.ServiceProxy("get_viz_options", GetVizOptions)

    def compute_property(self, *args):
        raise NotImplementedError()

    def should_keep(self, viz_options: VizOptions, *args):
        for name, filter_type in zip(viz_options.names, viz_options.filter_types):
            property_bool_value = self.compute_property(name, *args)
            if property_bool_value and filter_type == VizOptions.KEEP_ONLY_IF_FALSE:
                return False
            elif not property_bool_value and filter_type == VizOptions.KEEP_ONLY_IF_TRUE:
                return False
        return True

    def get(self):
        viz_options_req = GetVizOptionsRequest(names=self.filter_names)
        viz_options_res: GetVizOptionsResponse = self.get_viz_options_srv(viz_options_req)
        return viz_options_res.viz_options


class MyClassifierFilter(MyFilter):

    def __init__(self):
        super().__init__(['mistake', 'is_close'])

    def compute_property(self, name, example, predictions):
        is_close = bool(example['is_close'][1])
        binary_predictions = bool(predictions['probabilities'] > 0.5)
        if name == 'is_close':
            return is_close
        elif name == 'mistake':
            return binary_predictions != is_close


def visualize_iterative_classifier_adaption(ift_dir: pathlib.Path):
    log = load_hjson(ift_dir / 'logfile.hjson')
    planner_params_filename0 = ift_dir / 'planning_results' / 'iteration_0000_planning' / 'metadata.hjson'
    planner_params = load_hjson(planner_params_filename0)['planner_params']
    scenario = get_scenario(planner_params['scenario'])
    planning_iteration_dirs = ift_dir / 'planning_results'

    mode = 'all'

    iterations = []
    for k in log.keys():
        m = re.fullmatch(r"iteration (\d+)", k)
        if m:
            iteration_idx = int(m.group(1))
            iterations.append(iteration_idx)
    iterations = sorted(iterations)
    assert iterations[-1] == len(iterations) - 1

    rviz_stepper = RvizAnimationController(iterations)

    dataset_dir_for_viz = ift_dir / pathlib.Path(log['iteration 0']['classifier dataset']['new_dataset_dir'])
    dataset_for_viz = ClassifierDatasetLoader([dataset_dir_for_viz], load_true_states=True, scenario=scenario)
    metadata = dataset_for_viz.scenario_metadata
    state_metadata_keys = dataset_for_viz.state_metadata_keys
    predicted_state_keys = dataset_for_viz.predicted_state_keys
    action_keys = dataset_for_viz.action_keys

    goal_threshold = planner_params['goal_params']['threshold']

    f = MyClassifierFilter()

    classifier_cache = {}
    planning_data_cache = {}

    last_iteration_idx = 0
    while not rviz_stepper.done:
        iteration_idx = rviz_stepper.t()

        # get filter/view options from RViz gui?
        viz_options = f.get()

        # data is a list of tuples of (input dict, output dict)
        g = marker_index_generator(0)

        # clear the markers from the previous iteration
        if iteration_idx < last_iteration_idx or iteration_idx == 0 or not viz_options.accumulate:
            scenario.reset_viz()
        last_iteration_idx = iteration_idx

        if viz_options.accumulate:
            iterations_to_plot = range(iteration_idx + 1)
        else:
            iterations_to_plot = [iteration_idx]

        for i in iterations_to_plot:
            data, planning_iteration_data = get_data_cached(classifier_cache, ift_dir, i,
                                                            planning_data_cache,
                                                            planning_iteration_dirs,
                                                            scenario,
                                                            mode)

            goal = planning_iteration_data['goal']
            scenario.plot_goal_rviz(goal, goal_threshold)

            for j, (example, predictions) in enumerate(data):
                if not f.should_keep(viz_options, example, predictions):
                    continue

                scenario.plot_environment_rviz(example)
                pred = index_time_with_metadata(metadata, example, state_metadata_keys + predicted_state_keys, t=0)
                pred_next = index_time_with_metadata(metadata, example, state_metadata_keys + predicted_state_keys, t=1)
                action = index_time_with_metadata(metadata, example, action_keys, t=0)
                t0_marker_idx = next(g)
                t1_marker_idx = next(g)
                label = 'predicted'
                color = 'blue' if example['is_close'][1] else 'red'
                lightness_multiplier = i / len(iterations_to_plot) * 0.8 + 0.1
                color = adjust_lightness(color, lightness_multiplier)
                scenario.plot_state_rviz(pred, label=label, color=color, idx=t0_marker_idx, scale=0.2)
                scenario.plot_state_rviz(pred_next, label=label, color=color, idx=t1_marker_idx, scale=0.2)
                scenario.plot_action_rviz(pred, action, label=label, color=color, idx=t0_marker_idx, scale=1.0)

        rviz_stepper.step()


def get_data_cached(classifier_cache, ift_dir, iteration_idx, planning_data_cache, planning_iteration_dirs, scenario,
                    mode):
    if len(classifier_cache) > 50:
        last_entry = list(classifier_cache.keys())[-1]
        classifier_cache.pop(last_entry)
    if len(planning_data_cache) > 50:
        last_entry = list(planning_data_cache.keys())[-1]
        planning_data_cache.pop(last_entry)

    dataset_iteration_idx = iteration_idx
    classifier_iteration_idx = iteration_idx
    if iteration_idx in planning_data_cache:
        planning_iteration_data = planning_data_cache[iteration_idx]
    else:
        planning_iteration_dir = get_named_item_in_dir(planning_iteration_dirs, iteration_idx)
        planning_iteration_data = load_gzipped_pickle(next(planning_iteration_dir.glob("*.pkl.gz")))
        planning_data_cache[iteration_idx] = planning_iteration_data

    if dataset_iteration_idx in classifier_cache and classifier_iteration_idx in classifier_cache[
        dataset_iteration_idx]:
        data = classifier_cache[dataset_iteration_idx][classifier_iteration_idx]
    else:
        data = get_data(scenario, ift_dir, dataset_iteration_idx, classifier_iteration_idx, mode)
        if dataset_iteration_idx not in classifier_cache:
            classifier_cache[dataset_iteration_idx] = {}
        classifier_cache[dataset_iteration_idx][classifier_iteration_idx] = data

    return data, planning_iteration_data


def get_named_item_in_dir(directory, idx: int):
    for d in directory.iterdir():
        if re.match(f"[^\d]*0*{idx}[^\d]*$", d.name):
            return d


@ros_init.with_ros("viz_itr")
def main():
    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("ift_dir", type=pathlib.Path)

    args = parser.parse_args()

    visualize_iterative_classifier_adaption(**vars(args))


if __name__ == '__main__':
    main()
