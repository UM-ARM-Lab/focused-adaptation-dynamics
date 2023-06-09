from typing import Dict, List, Optional

import numpy as np
import torch

from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors, dict_of_tensors_to_sequence_of_dicts
from moonshine.torchify import torchify
from state_space_dynamics.torch_udnn import UDNN
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


def zeros_like(v):
    if isinstance(v, torch.Tensor):
        return torch.zeros_like(v)
    else:
        return np.zeros_like(v)


class TorchUDNNDynamicsWrapper:

    def __init__(self, checkpoint: str, scenario: Optional = None):
        self.model: UDNN = load_udnn_model_wrapper(checkpoint)
        self.model.cuda()
        self.model.with_joint_positions = 'joint_positions' in self.model.data_collection_params[
            'state_description'] or isinstance(scenario, BaseDualArmRopeScenario)
        self.model.eval()
        self.horizon = 2
        self.name = 'MDE'

        if scenario is not None:
            self.model.scenario = scenario

        self.data_collection_params = self.model.data_collection_params
        self.max_step_size = self.model.max_step_size

    def propagate(self, environment: Dict, start_state: Dict, actions: List[Dict]):
        return numpify(self.propagate_tf(environment, start_state, actions))

    def propagate_tf(self, environment: Dict, start_state: Dict, actions: List[Dict]):
        actions_dict = sequence_of_dicts_to_dict_of_tensors(actions)

        inputs = {}
        inputs.update(environment)
        inputs.update(add_batch(start_state))
        inputs.update(actions_dict)
        inputs = torchify(inputs, device=self.model.device)
        inputs['time_idx'] = torch.arange(len(actions), device=self.model.device)

        predicted_states_mean = remove_batch(self.model(add_batch(inputs)))
        predicted_states_stdevs = {}
        for k in self.model.state_keys:
            predicted_states_stdevs[k] = zeros_like(predicted_states_mean[k])

        predicted_states_mean_list = dict_of_tensors_to_sequence_of_dicts(predicted_states_mean)
        predicted_states_stdevs_list = dict_of_tensors_to_sequence_of_dicts(predicted_states_stdevs)

        return predicted_states_mean_list, predicted_states_stdevs_list
