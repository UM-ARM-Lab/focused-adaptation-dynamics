import pathlib
from typing import Dict, List, Tuple

import tensorflow as tf
from colorama import Fore

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_tensors
from shape_completion_training.model.filepath_tools import load_trial
from shape_completion_training.my_keras_model import MyKerasModel


class Ensemble:

    def __init__(self, model_dirs: List[pathlib.Path], batch_size: int, scenario: ExperimentScenario):
        representative_model_dir = model_dirs[0]
        _, self.hparams = load_trial(representative_model_dir.parent.absolute())

        self.scenario = scenario
        self.batch_size = batch_size
        self.data_collection_params = self.hparams['dynamics_dataset_hparams']['data_collection_params']
        self.states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        self.action_description = self.hparams['dynamics_dataset_hparams']['action_description']

        self.nets: List[MyKerasModel] = []
        for model_dir in model_dirs:
            net, ckpt = self.make_net_and_checkpoint(batch_size, scenario)
            manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=1)

            status = ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)
                status.assert_existing_objects_matched()
            else:
                raise RuntimeError("Failed to restore!!!")

            self.nets.append(net)

    def make_net_and_checkpoint(self, batch_size, scenario) -> Tuple[MyKerasModel, tf.train.Checkpoint]:
        raise NotImplementedError()

    def from_example(self, example, training: bool = False):
        """ This is the function that all other functions eventually call """
        if 'batch_size' not in example:
            example['batch_size'] = self.get_batch_size(example)

        outputs = [net(net.preprocess_no_gradient(example, training), training=training) for net in self.nets]
        outputs_dict = sequence_of_dicts_to_dict_of_tensors(outputs)
        mean = {state_key: tf.math.reduce_mean(outputs_dict[state_key], axis=0) for state_key in self.get_output_keys()}
        stdev = {state_key: tf.math.reduce_std(outputs_dict[state_key], axis=0) for state_key in self.get_output_keys()}
        all_stdevs = tf.concat(list(stdev.values()), axis=2)
        mean['stdev'] = tf.reduce_sum(all_stdevs, axis=-1, keepdims=True)
        return mean, stdev

    def get_batch_size(self, example: Dict):
        raise NotImplementedError()

    def get_output_keys(self):
        raise NotImplementedError()
