import pathlib
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import os
import torch.nn.functional as F
from torch.nn import Linear, Sequential
import matplotlib.pyplot as plt

from link_bot_pycommon.water_env_util import get_pour_type_and_error

plt.rcParams['font.size'] = 18

from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.numpify import numpify
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors, vector_to_dict
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset


def soft_mask(global_step, mask_threshold, error, k_global=1, k_base=0):
    assert k_base > 0.01
    return 1 - torch.sigmoid((k_global * global_step + k_base) * (error - mask_threshold))


class UDNN(pl.LightningModule):
    def __init__(self, with_joint_positions=False, **hparams):
        super().__init__()
        self.save_hyperparameters()

        datset_params = self.hparams['dataset_hparams']
        if 'data_collection_params' not in datset_params:
            self.data_collection_params = datset_params
        else:
            self.data_collection_params = datset_params['data_collection_params']
        self.data_collection_params['scenario_params']['run_flex'] = False
        self.scenario = get_scenario(self.hparams.scenario, params=self.data_collection_params['scenario_params'])
        self.dataset_state_description: Dict = self.data_collection_params['state_description']
        self.dataset_action_description: Dict = self.data_collection_params['action_description']
        self.state_keys = self.hparams.state_keys
        self.state_metadata_keys = self.hparams.state_metadata_keys
        self.state_description = {k: self.dataset_state_description[k] for k in self.hparams.state_keys}
        self.total_state_dim = sum([self.dataset_state_description[k] for k in self.hparams.state_keys])
        self.total_action_dim = sum([self.dataset_action_description[k] for k in self.hparams.action_keys])
        self.with_joint_positions = with_joint_positions
        self.max_step_size = self.data_collection_params.get('max_step_size', 0.01)  # default for current rope sim
        self.loss_scaling_by_key = self.hparams.get("loss_scaling_by_key", {})

        in_size = self.total_state_dim + self.total_action_dim
        if self.hparams.get("use_global_frame", False):
            in_size += self.total_state_dim
        fc_layer_size = None

        layers = []
        for layer_i, fc_layer_size in enumerate(self.hparams.fc_layer_sizes):
            layers.append(Linear(in_size, fc_layer_size))
            layers.append(torch.nn.ReLU())
            if self.hparams.get("use_batchnorm", False):
                layers.append(torch.nn.BatchNorm1d(fc_layer_size))
            if layer_i < len(self.hparams.fc_layer_sizes) - 1:
                if self.hparams.get('dropout_p', 0.0) > 0:
                    layers.append(torch.nn.Dropout(p=self.hparams.get('dropout_p', 0.0)))
                else:
                    assert False
            in_size = fc_layer_size
        layers.append(Linear(fc_layer_size, self.total_state_dim))

        self.mlp = Sequential(*layers)
        self.mae = torch.nn.L1Loss()

        if self.hparams.get('planning_mask', False):
            torch_ref_dataset = TorchDynamicsDataset(fetch_udnn_dataset(pathlib.Path('known_good_4')), mode='test')
            ref_actions_list = []
            for ref_traj in torch_ref_dataset:
                ref_s_0 = torch_ref_dataset.index_time(ref_traj, 0)
                ref_left_gripper_0 = ref_s_0['left_gripper']
                ref_right_gripper_0 = ref_s_0['right_gripper']
                ref_before = np.concatenate([ref_left_gripper_0, ref_right_gripper_0])

                ref_traj_len = len(ref_traj['time_idx'])
                for ref_t in range(ref_traj_len):
                    ref_s_t = torch_ref_dataset.index_time(ref_traj, ref_t)
                    ref_left_gripper_t = ref_s_t['left_gripper_position']
                    ref_right_gripper_t = ref_s_t['right_gripper_position']
                    ref_after = np.concatenate([ref_left_gripper_t, ref_right_gripper_t])
                    ref_actions = np.concatenate([ref_before, ref_after])
                    ref_actions_list.append(ref_actions)

                    ref_before = ref_after
            self.register_buffer("ref_actions", torch.tensor(ref_actions_list))

        self.fix_global_frame_bug = False
        self.test_errors = []

    def forward(self, inputs):
        actions = {k: inputs[k] for k in self.hparams.action_keys}
        input_sequence_length = actions[self.hparams.action_keys[0]].shape[1]
        s_0 = {k: inputs[k][:, 0] for k in self.hparams.state_keys}

        pred_states = [s_0]
        for t in range(input_sequence_length):
            s_t = pred_states[-1]
            action_t = {k: inputs[k][:, t] for k in self.hparams.action_keys}
            s_t_plus_1 = self.one_step_forward(action_t, s_t)

            pred_states.append(s_t_plus_1)

        pred_states_dict = sequence_of_dicts_to_dict_of_tensors(pred_states, axis=1)

        if self.with_joint_positions:
            # no need to do this during training, only during prediction/evaluation/testing
            inputs_np = numpify(inputs)
            inputs_np['batch_size'] = inputs['time_idx'].shape[0]
            _, joint_positions, joint_names = self.scenario.follow_jacobian_from_example(inputs_np,
                                                                                         j=self.scenario.robot.jacobian_follower)
            pred_states_dict['joint_positions'] = torchify(joint_positions).float()
            pred_states_dict['joint_names'] = joint_names

        return pred_states_dict

    def one_step_forward(self, action_t, s_t):
        REAL2SIM_OFFSET = torch.tensor([1, -0.5, .5]).to(self.device)

        local_action_t = self.scenario.put_action_local_frame(s_t, action_t)
        s_t_local = self.scenario.put_state_local_frame_torch(s_t)
        states_and_actions = list(s_t_local.values()) + list(local_action_t.values())
        if self.hparams.get("use_global_frame", False):
            if self.fix_global_frame_bug:
                rope_real_world_global_frame = s_t['rope'].reshape([-1, 25, 3])
                rope_sim_global_frame = (rope_real_world_global_frame + REAL2SIM_OFFSET).reshape([-1, 75])
                states_and_actions.append(rope_sim_global_frame)
                left_gripper_real_world_global_frame = s_t['left_gripper']
                left_gripper_sim_global_frame = left_gripper_real_world_global_frame + REAL2SIM_OFFSET
                states_and_actions.append(left_gripper_sim_global_frame)
                right_gripper_real_world_global_frame = s_t['right_gripper']
                right_gripper_sim_global_frame = right_gripper_real_world_global_frame + REAL2SIM_OFFSET
                states_and_actions.append(right_gripper_sim_global_frame)
            else:
                states_and_actions += list(s_t.values())
        z_t = torch.cat(states_and_actions, -1)
        z_t = self.mlp(z_t)
        delta_s_t = vector_to_dict(self.state_description, z_t, self.device)
        s_t_plus_1 = self.scenario.integrate_dynamics(s_t, delta_s_t)
        return s_t_plus_1

    def compute_batch_loss(self, inputs, outputs, use_mask: bool):
        batch_time_loss = compute_batch_time_loss(inputs, outputs, loss_scaling_by_key=self.loss_scaling_by_key)
        if use_mask:
            if self.hparams.get('iterative_lowest_error', False):
                mask_padded = self.low_error_mask(inputs, outputs)
                batch_time_loss = mask_padded * batch_time_loss
            elif self.hparams.get("low_initial_error", False):
                initial_model_outputs = self.initial_model.forward(inputs)
                mask_padded = self.low_error_mask(inputs, initial_model_outputs)
                batch_time_loss = mask_padded * batch_time_loss

        if 'time_mask' in inputs:
            time_mask = inputs['time_mask']
            batch_time_loss = time_mask * batch_time_loss

        batch_loss = batch_time_loss.sum(-1)

        return {
            'loss': batch_loss,
        }

    def low_error_mask(self, inputs, outputs, global_step=None):
        with torch.no_grad():
            error = self.scenario.classifier_distance_torch(inputs, outputs)

            if self.hparams.get("soft_masking", False):
                # FIXME: I think this should just be: low_error_mask = self.soft_mask(error, global_step=global_step)
                low_error_mask = self.soft_mask(error[:, :-1], global_step=global_step) * self.soft_mask(error[:, 1:],
                                                                                                         global_step=global_step)
            else:
                low_error_mask = error < self.hparams['mask_threshold']
                low_error_mask = torch.logical_and(low_error_mask[:, :-1], low_error_mask[:, 1:])

            mask = low_error_mask * (inputs["time_mask"][:,1:])
            mask = mask.float()
            mask_padded = F.pad(mask, [1, 0])

            if self.trainer:
                self.log("iterative mask mean", mask.mean())

        return mask_padded

    def soft_mask(self, error, global_step=None):
        if global_step is None:
            global_step = self.global_step
        low_error_mask = soft_mask(global_step, self.hparams['mask_threshold'], error, k_global = self.hparams.get("k_global", 1.0), k_base = self.hparams.get("k_base", 0))
        return low_error_mask

    def compute_loss(self, inputs, outputs, use_mask: bool):
        batch_losses = self.compute_batch_loss(inputs, outputs, use_mask)
        return {k: v.mean() for k, v in batch_losses.items()}

    def _add_noise_to_batch(self, batch):
        self.sigma = 0.001
        self.noise = torch.tensor(0).cuda()
        if self.training:
            for k in self.hparams.state_keys + self.hparams.action_keys:
                if isinstance(batch[k], torch.Tensor):
                    x = batch[k]
                    scale = self.sigma * x.detach() 
                    sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
                    batch[k] = x + sampled_noise

    def training_step(self, train_batch, batch_idx):
        #add noise to batch
        self._add_noise_to_batch(train_batch)
        outputs = self.forward(train_batch)
        if 'use_mask_train' in self.hparams:
            use_mask = self.hparams.get('use_mask_train', False)
        else:
            use_mask = self.hparams.get('use_meta_mask_train', False)
        losses = self.compute_loss(train_batch, outputs, use_mask)
        self.log('train_loss', losses['loss'])
        if batch_idx == 0:
            self.easy_accuracies = []
            self.easy_errors = []
            self.hard_accuracies = []
            self.hard_errors = []
        with torch.no_grad():
            error = self.scenario.classifier_distance_torch(train_batch, outputs)
            for k, y_pred in outputs.items():
                if 'time_mask' in train_batch:
                    time_mask = train_batch['time_mask'].bool()
                self.log(f'train_mae_{k}', self.mae(outputs[k][time_mask], train_batch[k][time_mask]))
            #Check accuracy over "pours"
            for sample_i in range(train_batch["target_volume"].shape[0]):
                pour_error, pour_type, sample_error = get_pour_type_and_error(train_batch, outputs, error,
                                                                              sample_i)
                if pour_type == "easy_pour":
                    self.easy_errors.append(sample_error)
                    self.easy_accuracies.append(pour_error)
                if pour_type == "hard_pour":
                    self.hard_errors.append(sample_error)
                    self.hard_accuracies.append(pour_error)
        self.log(f'train_easy_pour_error',np.mean(self.easy_errors))
        self.log(f'train_hard_pour_error',np.mean(self.hard_errors))
        return losses['loss']


    def on_train_epoch_end(self):
        viz_epochs = [0,10,20,50,100] #,20, 40]
        if "VIZ" in os.environ and self.current_epoch in viz_epochs : 
            if self.current_epoch == 0:
                self.fig, self.axs = plt.subplots(nrows = len(viz_epochs), sharex=True)
                self.axs[0].set_xlabel("Error")

            hist_range = (0, 1.5)
            alpha = 0.7
            viz_idx = viz_epochs.index(self.current_epoch)
            self.axs[viz_idx].hist(self.hard_errors, label="hard", bins=90, range=hist_range, alpha=alpha)
            self.axs[viz_idx].hist(self.easy_errors, label="easy", bins=90, range = hist_range, alpha=alpha)
            self.axs[viz_idx].set_title(f"epoch: {int(self.current_epoch)}")
            self.axs[viz_idx].set_xlim([0,1.5])
            self.axs[viz_idx].set_ylim([0,100])
            #axs[viz_idx].legend()
            if viz_idx == len(viz_epochs)-1:
                import ipdb; ipdb.set_trace()
                plt.tight_layout()
                plt.show()

    def validation_step(self, val_batch, batch_idx):
        val_udnn_outputs = self.forward(val_batch)
        if 'use_mask_val' in self.hparams:
            use_mask = self.hparams.get('use_mask_val', False)
        else:
            use_mask = self.hparams.get('use_meta_mask_val', False)
        val_losses = self.compute_loss(val_batch, val_udnn_outputs, use_mask)
        self.log('val_loss', val_losses['loss'])
        print(self.current_epoch, val_losses)
        return val_losses['loss']

    def test_step(self, test_batch, batch_idx):
        test_udnn_outputs = self.forward(test_batch)
        test_losses = self.compute_loss(test_batch, test_udnn_outputs, use_mask=False)
        test_losses['error'] = self.scenario.classifier_distance_torch(test_batch, test_udnn_outputs)
        self.log('test_error', test_losses['error'])
        self.log('test_loss', test_losses['loss'])
        self.test_errors.append(test_losses['error'])

        return test_losses

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0)

    def state_dict(self, *args, **kwargs):
        return self.state_dict_without_initial_model()

    def state_dict_without_initial_model(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)
        out_d = {}
        for k, v in d.items():
            if not k.startswith('initial_model'):
                out_d[k] = v
        return out_d

    def on_load_checkpoint(self, checkpoint: Dict):
        if self.hparams.get('low_initial_error', False):
            from copy import deepcopy
            initial_model_hparams = deepcopy(self.hparams)
            initial_model_hparams.pop("low_initial_error")
            self.initial_model = UDNN(**initial_model_hparams)
            self.initial_model.load_state_dict(checkpoint["state_dict"])

    def load_state_dict(self, state_dict, strict: bool = False):
        self.load_state_dict_ignore_missing_initial_model(state_dict)

    def load_state_dict_ignore_missing_initial_model(self, state_dict):
        super().load_state_dict(state_dict, strict=False)


def compute_batch_time_loss(inputs, outputs, loss_scaling_by_key={}):
    loss_by_key = []
    for k, y_pred in outputs.items():
        y_true = inputs[k]
        loss_scaling = loss_scaling_by_key[k] if k in loss_scaling_by_key else 1

        # mean over time and state dim but not batch, not yet.
        loss = loss_scaling * (y_true - y_pred).square().mean(-1)
        loss_by_key.append(loss)
    batch_time_loss = torch.stack(loss_by_key).mean(0)
    return batch_time_loss
