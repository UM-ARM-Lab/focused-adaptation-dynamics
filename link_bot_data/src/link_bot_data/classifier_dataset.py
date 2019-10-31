import json
import pathlib
import random
from typing import Optional, List

import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_data.video_prediction_dataset_utils import float_feature
from link_bot_planning.params import LocalEnvParams, EnvParams, PlannerParams


class ClassifierDataset:

    def __init__(self,
                 dataset_dir: pathlib.Path,
                 is_labeled: bool = False,
                 ):
        self.is_labeled = is_labeled
        self.dataset_dir = dataset_dir
        dataset_hparams_filename = dataset_dir / 'hparams.json'
        if not self.is_labeled and 'labeled' in str(dataset_dir):
            print(Fore.YELLOW + "I noticed 'labeled' in the dataset path, so I will attempt to load labels" + Fore.RESET)
            self.is_labeled = True
        self.hparams = json.load(open(str(dataset_hparams_filename), 'r'))
        self.hparams['local_env_params'] = LocalEnvParams.from_json(self.hparams['local_env_params'])
        self.hparams['env_params'] = EnvParams.from_json(self.hparams['env_params'])
        self.hparams['planner_params'] = PlannerParams.from_json(self.hparams['planner_params'])

    def parser(self, local_env_shape, n_state, n_action):
        def _parser(serialized_example):
            features = {
                'actual_local_env/env': tf.FixedLenFeature(local_env_shape, tf.float32),
                'actual_local_env/extent': tf.FixedLenFeature([4], tf.float32),
                'actual_local_env/origin': tf.FixedLenFeature([2], tf.float32),
                'planned_local_env/env': tf.FixedLenFeature(local_env_shape, tf.float32),
                'planned_local_env/extent': tf.FixedLenFeature([4], tf.float32),
                'planned_local_env/origin': tf.FixedLenFeature([2], tf.float32),
                'res': tf.FixedLenFeature([1], tf.float32),
                'w_m': tf.FixedLenFeature([1], tf.float32),
                'h_m': tf.FixedLenFeature([1], tf.float32),
                'state': tf.FixedLenFeature([n_state], tf.float32),
                'next_state': tf.FixedLenFeature([n_state], tf.float32),
                'action': tf.FixedLenFeature([n_action], tf.float32),
                'planned_state': tf.FixedLenFeature([n_state], tf.float32),
                'planned_next_state': tf.FixedLenFeature([n_state], tf.float32),
            }
            if self.is_labeled:
                features['label'] = tf.FixedLenFeature([1], tf.float32)
            features = tf.parse_single_example(serialized_example, features=features)

            return features

        return _parser

    @classmethod
    def make_features_dict(cls,
                           actual_local_env,
                           actual_local_env_extent,
                           actual_local_env_origin,
                           planned_local_env,
                           planned_local_env_extent,
                           planned_local_env_origin,
                           y_height,
                           x_width,
                           res,
                           state,
                           next_state,
                           action,
                           planned_state,
                           planned_next_state,
                           traj_idx,
                           time_idx):
        features = {
            'actual_local_env/env': float_feature(actual_local_env.flatten()),
            'actual_local_env/extent': float_feature(np.array(actual_local_env_extent)),
            'actual_local_env/origin': float_feature(np.array(actual_local_env_origin)),
            'planned_local_env/env': float_feature(planned_local_env.flatten()),
            'planned_local_env/extent': float_feature(np.array(planned_local_env_extent)),
            'planned_local_env/origin': float_feature(np.array(planned_local_env_origin)),
            'res': float_feature(np.array([res])),
            'w_m': float_feature(np.array([y_height])),
            'h_m': float_feature(np.array([x_width])),
            'state': float_feature(state),
            'next_state': float_feature(next_state),
            'action': float_feature(action),
            'planned_state': float_feature(planned_state),
            'planned_next_state': float_feature(planned_next_state),
            'traj_idx': float_feature(np.array([traj_idx])),
            'time_idx': float_feature(np.array([time_idx])),
        }
        return features

    @classmethod
    def make_serialized_example(cls, *args):
        features = cls.make_features_dict(*args)
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        example = example_proto.SerializeToString()
        return example

    def get_dataset(self,
                    mode: str,
                    num_epochs: int,
                    batch_size: Optional[int],
                    shuffle: bool = True,
                    seed: int = 0):
        filenames = [str(filename) for filename in self.dataset_dir.glob("{}/*.tfrecords".format(mode))]
        return self.get_dataset_from_records(filenames, num_epochs, batch_size, shuffle, seed)

    def get_dataset_all_modes(self,
                              num_epochs: int,
                              batch_size: Optional[int],
                              shuffle: bool = True,
                              seed: int = 0):
        train_filenames = [str(filename) for filename in self.dataset_dir.glob("{}/*.tfrecords".format('train'))]
        test_filenames = [str(filename) for filename in self.dataset_dir.glob("{}/*.tfrecords".format('test'))]
        val_filenames = [str(filename) for filename in self.dataset_dir.glob("{}/*.tfrecords".format('val'))]

        all_filenames = train_filenames
        all_filenames.extend(test_filenames)
        all_filenames.extend(val_filenames)

        return self.get_dataset_from_records(all_filenames, num_epochs, batch_size, shuffle, seed)

    def get_dataset_from_records(self,
                                 records: List[str],
                                 num_epochs: int,
                                 batch_size: Optional[int],
                                 shuffle: bool = True,
                                 seed: int = 0):
        compression_type = self.hparams['compression_type']
        local_env_rows = int(self.hparams['local_env_params'].h_rows)
        local_env_cols = int(self.hparams['local_env_params'].w_cols)
        local_env_shape = [local_env_rows, local_env_cols]

        n_state = int(self.hparams['n_state'])
        n_action = int(self.hparams['n_action'])

        if shuffle:
            random.shuffle(records)

        dataset = tf.data.TFRecordDataset(records, buffer_size=8 * 1024 * 1024, compression_type=compression_type)

        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=num_epochs, seed=seed))
        else:
            dataset = dataset.repeat(num_epochs)

        dataset = dataset.map(self.parser(local_env_shape, n_state, n_action))
        if batch_size is not None and batch_size > 0:
            dataset = dataset.batch(batch_size)

        return dataset
