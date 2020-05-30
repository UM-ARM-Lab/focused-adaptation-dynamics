#!/usr/bin/env python
import argparse
import json
import logging
import pathlib
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_classifiers.analysis_utils import predict_and_execute, load_models
from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon import ros_pycommon
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_sdf_utils import env_from_occupancy_data
from link_bot_pycommon.ros_pycommon import get_states_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, listify, numpify

limit_gpu_mem(4)


def save_data(basedir, environment, actuals, predictions, accept_probabilities, random_actions):
    data = {
        'environment': environment,
        'actuals': actuals,
        'predictions': predictions,
        'accept_probabilities': accept_probabilities,
        'random_actions': random_actions,
    }
    filename = basedir / 'saved_data.json'
    print(Fore.GREEN + f'saving {filename.as_posix()}' + Fore.RESET)
    json.dump(listify(data), filename.open("w"))


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()
    test_config_parser = subparsers.add_parser('test_config')
    test_config_parser.add_argument('test_config', help="json file describing the test", type=pathlib.Path)
    test_config_parser.add_argument('--n-actions-sampled', type=int, default=100)
    test_config_parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path, nargs='+')
    test_config_parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    test_config_parser.set_defaults(func=test_config)
    load_parser = subparsers.add_parser('load')
    load_parser.add_argument('load_from', help="json file with previously generated results", type=pathlib.Path)
    load_parser.set_defaults(func=load)

    np.set_printoptions(suppress=True, precision=3)
    np.random.seed(0)
    tf.random.set_seed(0)
    rospy.init_node("recovery_check")
    tf.get_logger().setLevel(logging.ERROR)

    args = parser.parse_args()
    # args.fwd_model_dirs = [pathlib.Path(f"./ss_log_dir/tf2_rope/{i}") for i in range(8)]
    # args.classifier_model_dir = pathlib.Path('log_data/rope_2_seq/May_24_01-12-08_617a0bee2a')
    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


def test_config(args):
    now = time.time()
    basedir = pathlib.Path(f"results/recovery_check/{int(now)}")
    basedir.mkdir(exist_ok=True)

    test_config = json.load(args.test_config.open("r"))
    action_sequence_length = 6
    random_actions = sample_actions(args.n_actions_sampled, action_sequence_length)
    start_configs = [test_config['start_config']] * args.n_actions_sampled
    results = predict_and_execute(args.classifier_model_dir, args.fwd_model_dir, test_config, start_configs, random_actions)
    fwd_model, classifier_model, environment, actuals, predictions, accept_probabilities = results

    save_data(basedir, environment, actuals, predictions, accept_probabilities, random_actions)

    compare_predictions_to_actual(basedir=basedir,
                                  classifier=classifier_model,
                                  environment=environment,
                                  random_actions=random_actions,
                                  predictions=predictions,
                                  actuals=actuals,
                                  accepts_probabilities=accept_probabilities)


def load(args):
    saved_data = json.load(args.load_from.open("r"))
    environment = numpify(saved_data['environment'])
    actuals = [numpify(a_i) for a_i in saved_data['actuals']]
    predictions = [numpify(p_i) for p_i in saved_data['predictions']]
    random_actions = numpify(saved_data['random_actions'])
    accept_probabilities = numpify(saved_data['accept_probabilities'])
    classifier_model, fwd_model = load_models(classifier_model_dir=args.classifier_model_dir, fwd_model_dir=args.fwd_model_dir)

    basedir = args.load_from.parent
    compare_predictions_to_actual(basedir=basedir,
                                  classifier=classifier_model,
                                  environment=environment,
                                  random_actions=random_actions,
                                  predictions=predictions,
                                  actuals=actuals,
                                  accepts_probabilities=accept_probabilities)


def compare_predictions_to_actual(basedir: pathlib.Path,
                                  classifier: BaseConstraintChecker,
                                  environment: Dict,
                                  random_actions,
                                  predictions: List,
                                  actuals: List,
                                  accepts_probabilities
                                  ):
    labeling_params = classifier.model_hparams['classifier_dataset_hparams']['labeling_params']
    labeling_params['threshold'] = 0.05
    key = labeling_params['state_key']
    all_predictions_are_far = []
    all_predictions_are_rejected = []
    for i, (prediction, actual, actions, accept_probabilities) in enumerate(
            zip(predictions, actuals, random_actions, accepts_probabilities)):
        prediction_seq = dict_of_sequences_to_sequence_of_dicts(prediction)
        actual_seq = dict_of_sequences_to_sequence_of_dicts(actual)
        all_prediction_is_close = np.linalg.norm(prediction[key] - actual[key], axis=1) < labeling_params['threshold']
        # [1:] because start state will match perfectly
        last_prediction_is_close = all_prediction_is_close[-1]
        prediction_is_far = np.logical_not(last_prediction_is_close)
        prediction_is_rejected = accept_probabilities[-1] < 0.5
        classifier_says = 'reject' if prediction_is_rejected else 'accept'
        print(
            f"action sequence {i}, final prediction is close to ground truth {last_prediction_is_close}, classifier says {classifier_says}")
        all_predictions_are_far.append(prediction_is_far)
        all_predictions_are_rejected.append(prediction_is_rejected)
        anim = classifier.scenario.animate_predictions(environment=environment,
                                                       actions=actions,
                                                       actual=actual_seq,
                                                       predictions=prediction_seq,
                                                       labels=all_prediction_is_close,
                                                       accept_probabilities=accept_probabilities)

        outfilename = basedir / f'action_{i}.gif'
        anim.save(outfilename, writer='imagemagick', dpi=200)
        plt.close()

    if np.all(all_predictions_are_rejected):
        print("needs recovery!")


def get_state_and_environment(classifier_model, scenario, service_provider):
    full_env_data = ros_pycommon.get_occupancy_data(env_w_m=classifier_model.full_env_params.w,
                                                    env_h_m=classifier_model.full_env_params.h,
                                                    res=classifier_model.full_env_params.res,
                                                    service_provider=service_provider,
                                                    robot_name=scenario.robot_name())
    environment = env_from_occupancy_data(full_env_data)
    state_dict = get_states_dict(service_provider)
    return environment, state_dict


def sample_actions(n_samples, horizon):
    return tf.random.uniform(shape=[n_samples, horizon, 2], minval=-0.15, maxval=0.15)


if __name__ == '__main__':
    main()
