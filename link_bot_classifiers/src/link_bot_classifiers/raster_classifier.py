#!/usr/bin/env python
from __future__ import print_function

import json
import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from colorama import Fore
from tensorflow import keras

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_planning.params import LocalEnvParams
from moonshine.numpy_utils import add_batch
from moonshine.raster_points_layer import make_transition_image, make_traj_images
from moonshine.tensorflow_train_test_loop import MyKerasModel


class RasterClassifier(MyKerasModel):

    def __init__(self, hparams: Dict, batch_size: int):
        super().__init__(hparams, batch_size)

        self.dynamics_dataset_hparams = self.hparams['classifier_dataset_hparams']['fwd_model_hparams'][
            'dynamics_dataset_hparams']
        self.n_action = self.dynamics_dataset_hparams['n_action']
        self.batch_size = batch_size

        self.states_keys = self.hparams['states_keys']

        self.local_env_params = LocalEnvParams.from_json(self.dynamics_dataset_hparams['local_env_params'])

        self.conv_layers = []
        self.pool_layers = []
        for n_filters, kernel_size in self.hparams['conv_filters']:
            conv = layers.Conv2D(n_filters,
                                 kernel_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 activity_regularizer=keras.regularizers.l1(self.hparams['activity_reg']))
            pool = layers.MaxPool2D(2)
            self.conv_layers.append(conv)
            self.pool_layers.append(pool)

        self.conv_flatten = layers.Flatten()
        if self.hparams['batch_norm']:
            self.batch_norm = layers.BatchNormalization()

        self.dense_layers = []
        self.dropout_layers = []
        for hidden_size in self.hparams['fc_layer_sizes']:
            dropout = layers.Dropout(rate=self.hparams['dropout_rate'])
            dense = layers.Dense(hidden_size,
                                 activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(self.hparams['kernel_reg']),
                                 bias_regularizer=keras.regularizers.l2(self.hparams['bias_reg']),
                                 activity_regularizer=keras.regularizers.l1(self.hparams['activity_reg']))
            self.dropout_layers.append(dropout)
            self.dense_layers.append(dense)

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def _conv(self, image):
        # feed into a CNN
        conv_z = image
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            conv_h = conv_layer(conv_z)
            conv_z = pool_layer(conv_h)
        out_conv_z = conv_z

        return out_conv_z

    def call(self, input_dict: Dict, training=None, mask=None):
        # Choose what key to use, so depending on how the model was trained it will expect a transition_image or trajectory_image
        image = input_dict[self.hparams['image_key']]
        action = input_dict['action']
        out_conv_z = self._conv(image)
        conv_output = self.conv_flatten(out_conv_z)

        if self.hparams['mixed']:
            concat_args = [conv_output, action]
            for state_key in self.states_keys:
                planned_state_key = 'planned_state/{}'.format(state_key)
                planned_state_key_next = 'planned_state_next/{}'.format(state_key)
                state = input_dict[planned_state_key]
                next_state = input_dict[planned_state_key_next]
                concat_args.append(state)
                concat_args.append(next_state)
            conv_output = tf.concat(concat_args, axis=1)

        if self.hparams['batch_norm']:
            conv_output = self.batch_norm(conv_output)

        z = conv_output
        for dropout_layer, dense_layer in zip(self.dropout_layers, self.dense_layers):
            h = dropout_layer(z)
            z = dense_layer(h)
        out_h = z

        accept_probability = self.output_layer(out_h)
        return accept_probability


class RasterClassifierWrapper(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(scenario)
        model_hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(model_hparams_file.open('r'))
        self.net = RasterClassifier(hparams=self.model_hparams, batch_size=batch_size)
        self.local_env_params = self.net.local_env_params
        self.ckpt = tf.train.Checkpoint(net=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=1)
        if self.manager.latest_checkpoint:
            print(Fore.CYAN + "Restored from {}".format(self.manager.latest_checkpoint) + Fore.RESET)
        self.ckpt.restore(self.manager.latest_checkpoint)

    def check_transition(self,
                         full_env,
                         full_env_origin,
                         res,
                         states_sequence: List[Dict],
                         actions,
                         ) -> tf.Tensor:
        states_i = states_sequence[-2]
        action_i = actions[-1]
        states_i_plus_1 = states_sequence[-1]

        action_in_image = self.model_hparams['action_in_image']
        batched_inputs = add_batch(full_env, full_env_origin, res, states_i, action_i, states_i_plus_1)
        image = make_transition_image(*batched_inputs,
                                      scenario=self.scenario,
                                      local_env_h=self.local_env_params.h_rows,
                                      local_env_w=self.local_env_params.w_cols,
                                      action_in_image=action_in_image)[0]
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        net_inputs = self.net_inputs(action_i, states_i, states_i_plus_1)
        net_inputs['transition_image'] = image

        accept_probability = self.net(add_batch(net_inputs))[0, 0]
        return accept_probability

    def check_trajectory(self,
                         full_env: np.ndarray,
                         full_env_origin: np.ndarray,
                         res: float,
                         states_sequence: List[Dict],
                         actions: tf.Variable) -> tf.Tensor:
        # Get state states/action for just the transition, which we also feed into the classifier
        action_i = actions[-1]
        states_i = states_sequence[-2]
        states_i_plus_1 = states_sequence[1]

        batched_inputs = add_batch(full_env.data, full_env_origin, res, states_sequence)
        image = make_traj_images(*batched_inputs)[0]

        net_inputs = self.net_inputs(action_i, states_i, states_i_plus_1)
        net_inputs['trajectory_image'] = image

        accept_probability = self.net(add_batch(net_inputs))[0, 0]
        return accept_probability

    def check_constraint_differentiable(self,
                                        full_env: np.ndarray,
                                        full_env_origin: np.ndarray,
                                        res: float,
                                        states_sequence: List[Dict],
                                        actions: tf.Variable) -> tf.Tensor:
        image_key = self.model_hparams['image_key']
        if image_key == 'transition_image':
            return self.check_transition(full_env=full_env,
                                         full_env_origin=full_env_origin,
                                         res=res,
                                         states_sequence=states_sequence,
                                         actions=actions)
        elif image_key == 'trajectory_image':
            return self.check_trajectory(full_env, full_env_origin, res, states_sequence, actions)
        else:
            raise ValueError('invalid image_key')

    def check_constraint(self,
                         full_env: np.ndarray,
                         full_env_origin: np.ndarray,
                         res: float,
                         states: List[Dict],
                         actions: np.ndarray) -> float:
        actions = tf.Variable(actions, dtype=tf.float32, name="actions")
        prediction = self.check_constraint_differentiable(full_env,
                                                          full_env_origin,
                                                          res,
                                                          states,
                                                          actions)
        return prediction.numpy()

    def net_inputs(self, action_i, states_i, states_i_plus_1):
        net_inputs = {
            'action': tf.convert_to_tensor(action_i, tf.float32),
        }

        for state_key in self.net.states_keys:
            planned_state_key = 'planned_state/{}'.format(state_key)
            planned_state_key_next = 'planned_state_next/{}'.format(state_key)
            net_inputs[planned_state_key] = tf.convert_to_tensor(states_i[state_key], tf.float32)
            net_inputs[planned_state_key_next] = tf.convert_to_tensor(states_i_plus_1[state_key], tf.float32)

        return net_inputs


model = RasterClassifierWrapper
