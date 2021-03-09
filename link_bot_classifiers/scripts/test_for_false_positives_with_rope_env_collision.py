#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import Dict

import colorama
import numpy as np
import tensorflow as tf
from progressbar import progressbar

import rospy
from arc_utilities import ros_init
from link_bot_classifiers import classifier_utils
from link_bot_classifiers.points_collision_checker import PointsCollisionChecker
from link_bot_data import base_dataset
from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.dataset_utils import remove_predicted, deserialize_scene_msg
from link_bot_data.visualization import init_viz_env
from link_bot_planning.results_utils import print_percentage
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from link_bot_pycommon.serialization import dump_gzipped_pickle
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine import filepath_tools
from moonshine.indexing import index_dict_of_batched_tensors_tf
from moonshine.my_keras_model import MyKerasModel
from std_msgs.msg import Float32


@ros_init.with_ros("test_for_fp")
def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    np.set_printoptions(linewidth=250, precision=4, suppress=True)
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val', 'all'], default='val')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--take', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--use-gt-rope', action='store_true')
    parser.add_argument('--only-fp', action='store_true')
    parser.add_argument('--only-fn', action='store_true')
    parser.add_argument('--only-mistakes', action='store_true')
    parser.add_argument('--only-starts-close', action='store_true')
    parser.add_argument('--only-starts-far', action='store_true')
    parser.add_argument('--only-tn', action='store_true')
    parser.add_argument('--only-predicted-not-in-collision', action='store_true')
    parser.add_argument('--only-predicted-in-collision', action='store_true')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--no-viz', action='store_true')
    parser.add_argument('--start-at', type=int, default=0)

    args = parser.parse_args()

    traj_idx_pub_ = rospy.Publisher("traj_idx_viz", Float32, queue_size=10)

    ###############
    # Model
    ###############
    trials_directory = pathlib.Path('trials').absolute()
    trial_path = args.checkpoint.parent.absolute()
    _, params = filepath_tools.create_or_load_trial(trial_path=trial_path,
                                                    trials_directory=trials_directory)

    ###############
    # Dataset
    ###############
    dataset = ClassifierDatasetLoader(args.dataset_dirs,
                                      load_true_states=True,
                                      use_gt_rope=args.use_gt_rope,
                                      threshold=args.threshold)
    tf_dataset = dataset.get_datasets(mode=args.mode)
    scenario = dataset.scenario

    ###############
    # Evaluate
    ###############
    tf_dataset = tf_dataset.batch(args.batch_size, drop_remainder=True)
    tf_dataset = tf_dataset.take(args.take)

    model = classifier_utils.load_generic_model(args.checkpoint)
    assert len(model.nets) == 1
    net: MyKerasModel = model.nets[0]

    cc = PointsCollisionChecker(pathlib.Path('trials/cc_baseline/none'), scenario)

    fp = 0
    fn = 0
    labeled_0 = 0
    predicted_in_collision = 0
    predicted_not_in_collision = 0
    predicted_in_collision_labeled_0 = 0
    not_predicted_in_collision_labeled_0 = 0
    predicted_in_collision_labeled_1 = 0
    not_predicted_in_collision_labeled_1 = 0
    predicted_in_collision_fp = 0
    not_predicted_in_collision_fp = 0
    predicted_in_collision_fn = 0
    count = 0
    n_correct = 0
    metrics = net.create_metrics()

    # for batch_idx, example in enumerate(tf_dataset):
    for batch_idx, example in enumerate(progressbar(tf_dataset, widgets=base_dataset.widgets)):

        if batch_idx < args.start_at:
            continue

        example_for_constr = example.copy()
        example_for_constr.update(dataset.batch_metadata)

        predictions, _ = model.check_constraint_from_example(example_for_constr, training=False)

        deserialize_scene_msg(example)
        example.pop("batch_size")

        starts_close = example['is_close'][:, 0]
        is_close = tf.expand_dims(example['is_close'][:, 1:], axis=2)

        probabilities = predictions['probabilities']

        is_predicted_close = tf.squeeze(tf.squeeze(probabilities > 0.5, axis=-1), axis=-1)
        is_close = tf.squeeze(tf.squeeze(tf.cast(is_close, tf.bool), axis=-1), axis=-1)
        classifier_is_correct = tf.equal(is_predicted_close, is_close)
        is_tn = tf.logical_and(tf.logical_not(is_close), tf.logical_not(is_predicted_close))
        is_fp = tf.logical_and(tf.logical_not(is_close), is_predicted_close)
        is_fn = tf.logical_and(is_close, tf.logical_not(is_predicted_close))

        # iterate over each example in batch, it's easier to visualize and compute some metrics this way
        for b in range(args.batch_size):
            example_b = index_dict_of_batched_tensors_tf(example, b)

            example_b_no_ps = index_dict_of_batched_tensors_tf(example, b, keep_dims=True)
            example_b_no_ps.pop("scene_msg")
            example_b_no_ps['batch_size'] = 1
            example_b_no_ps['time'] = 2
            for k in ['classifier_end_t',
                      'classifier_start_t',
                      'world_to_rgb_optical_frame',
                      'kinect_pose',
                      'kinect_params']:
                example_b_no_ps.pop(k)

            example_b_pred = {}
            example_b_pred.update(example_b)
            example_b_pred.update({remove_predicted(k): example_b[k] for k in dataset.predicted_state_keys})

            rope_points_not_in_collision, _ = cc.check_constraint_from_example(example_b_pred)
            rope_points_not_in_collision = rope_points_not_in_collision[0]
            rope_points_in_collision = not rope_points_not_in_collision

            # filter examples
            if args.only_starts_close:
                if not starts_close[b]:
                    continue

            if args.only_starts_far:
                if starts_close[b]:
                    continue

            if args.only_predicted_not_in_collision:
                if not rope_points_not_in_collision:
                    continue
            if args.only_predicted_in_collision:
                if rope_points_not_in_collision:
                    continue

            if args.only_mistakes:
                if classifier_is_correct[b]:
                    continue

            if args.only_fn:
                if not is_fn[b]:
                    continue

            if args.only_fp:
                if not is_fp[b]:
                    continue

            if args.only_tn:
                if not is_tn[b]:
                    continue
            # end filter examples

            # update metrics
            net.val_step(example_b_no_ps, metrics)

            count += 1
            if not is_close[b]:
                labeled_0 += 1
            if rope_points_not_in_collision:
                predicted_not_in_collision += 1
                if is_close[b]:
                    not_predicted_in_collision_labeled_1 += 1
                else:
                    not_predicted_in_collision_labeled_0 += 1
                if is_fp[b]:
                    not_predicted_in_collision_fp += 1
            if rope_points_in_collision:
                predicted_in_collision += 1
                if is_close[b]:
                    predicted_in_collision_labeled_1 += 1
                else:
                    predicted_in_collision_labeled_0 += 1
                if is_fp[b]:
                    predicted_in_collision_fp += 1
                if is_fn[b]:
                    predicted_in_collision_fn += 1
            if is_fp[b]:
                fp += 1
            if is_fn[b]:
                fn += 1
            if classifier_is_correct[b]:
                n_correct += 1
            # end update metrics

            # visualize
            if not args.no_viz:
                def _custom_viz_t(scenario: ScenarioWithVisualization, e: Dict, t: int):
                    if t > 0:
                        accept_probability_t = predictions['probabilities'][b, t - 1, 0].numpy()
                    else:
                        accept_probability_t = -999
                    scenario.plot_accept_probability(accept_probability_t)

                    traj_idx_msg = Float32()
                    traj_idx_msg.data = batch_idx * args.batch_size + b
                    traj_idx_pub_.publish(traj_idx_msg)

                anim = RvizAnimation(scenario=scenario,
                                     n_time_steps=dataset.horizon,
                                     init_funcs=[init_viz_env,
                                                 dataset.init_viz_action(),
                                                 ],
                                     t_funcs=[_custom_viz_t,
                                              dataset.classifier_transition_viz_t(),
                                              ExperimentScenario.plot_stdev_t,
                                              init_viz_env,
                                              ])

                dump_gzipped_pickle(example_b, pathlib.Path('debugging.pkl.gzip'))
                anim.play(example_b)
            # end visualize

    n_incorrect = count - n_correct

    print(args.dataset_dirs[0].as_posix())
    print(args.checkpoint.as_posix())
    print_percentage("% labeled 0", labeled_0, count)
    print_percentage("% correct (accuracy)", n_correct, count)
    print_percentage('% FP', fp, count)
    print_percentage('% FN', fn, count)
    print_percentage('% FP / mistakes', fp, n_incorrect)
    print_percentage('% FN / mistakes', fn, n_incorrect)
    print_percentage('% FP that are in collision',
                     predicted_in_collision_fp, fp)
    print_percentage('% FP that are in not collision',
                     not_predicted_in_collision_fp, fp)
    print_percentage('% predicted state is in collision',
                     predicted_in_collision, count)
    print_percentage('% predicted state is in collision and the label is 0 ',
                     predicted_in_collision_labeled_0, predicted_in_collision)
    print_percentage('% false positives and predicted state is in collision',
                     predicted_in_collision_fp, predicted_in_collision_labeled_0)
    print_percentage('% false positives and predicted state is not in collision',
                     not_predicted_in_collision_fp, not_predicted_in_collision_labeled_0)
    print_percentage('% false negatives and predicted state is in collision',
                     predicted_in_collision_fn, predicted_in_collision_labeled_1)

    print("NOTE: the following metrics are not affected some of the command line flags")
    for metric_name, metric in metrics.items():
        print(f"{metric_name:80s} {metric.result().numpy().squeeze() * 100:.2f}")


if __name__ == '__main__':
    main()
