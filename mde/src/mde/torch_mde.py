import pathlib
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

import rospy
from link_bot_data.dataset_utils import add_predicted_hack, add_predicted
from link_bot_data.local_env_helper import LocalEnvHelper
from link_bot_data.visualization import DebuggingViz
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.grid_utils_np import environment_to_vg_msg
from link_bot_pycommon.load_wandb_model import load_model_artifact
from moonshine import get_local_environment_torch
from moonshine.make_voxelgrid_inputs_torch import VoxelgridInfo
from moonshine.res3d import Res3D
from moonshine.robot_points_torch import RobotVoxelgridInfo
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.torch_utils import sequence_of_dicts_to_dict_of_tensors
from moonshine.torchify import torchify

from link_bot_pycommon.water_env_util import get_pour_type_and_error


def debug_vgs():
    return rospy.get_param("DEBUG_VG", False)


class MDE(pl.LightningModule):

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        datset_params = self.hparams['dataset_hparams']
        data_collection_params = datset_params['data_collection_params']
        self.scenario = get_scenario(self.hparams.scenario, params=data_collection_params['scenario_params'])

        self.local_env_h_rows = self.hparams['local_env_h_rows']
        self.local_env_w_cols = self.hparams['local_env_w_cols']
        self.local_env_c_channels = self.hparams['local_env_c_channels']
        self._c1 = self.hparams.get("c1", 3)
        self._c2 = self.hparams.get("c2", 1)
        self.in_channels = self.hparams.get('in_channels', 5)
        self.include_robot_geometry = self.hparams.get('include_robot_geometry', True)
        self.point_state_keys_pred = [add_predicted_hack(k) for k in self.hparams['point_state_keys']]

        conv_layers = []
        in_channels = self.in_channels
        if self.hparams.get("new_pooling", False):
            assert (len(self.hparams['conv_filters']) == len(self.hparams['new_pooling']))
            for i, ((out_channels, kernel_size), pooling) in enumerate(
                    zip(self.hparams['conv_filters'], self.hparams['new_pooling'])):
                if self.hparams.get("use_res3d", False) and i > 0:
                    conv_layers.append(Res3D(in_channels, out_channels, kernel_size))
                else:
                    conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size))
                    conv_layers.append(nn.LeakyReLU())
                if self.hparams.get("use_batchnorm", False):
                    conv_layers.append(nn.BatchNorm3d())
                conv_layers.append(nn.MaxPool3d(pooling))
                in_channels = out_channels
        else:
            for i, (out_channels, kernel_size) in enumerate(self.hparams['conv_filters']):
                if self.hparams.get("use_res3d", False) and i > 0:
                    conv_layers.append(Res3D(in_channels, out_channels, kernel_size))
                else:
                    conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size))
                    conv_layers.append(nn.LeakyReLU())
                if self.hparams.get("use_batchnorm", False):
                    conv_layers.append(nn.BatchNorm3d(out_channels))
                conv_layers.append(nn.MaxPool3d(self.hparams['pooling']))
                in_channels = out_channels

        fc_layers = []
        state_desc = data_collection_params['state_description']
        action_desc = data_collection_params['action_description']
        state_size = sum([state_desc[k] for k in self.hparams.state_keys])
        action_size = sum([action_desc[k] for k in self.hparams.action_keys])

        if self.hparams.get("conv_out_size", None):
            conv_out_size = self.hparams["conv_out_size"]
        else:
            conv_out_size = int(self.hparams['conv_filters'][-1][0] * np.prod(self.hparams['conv_filters'][-1][1]))

        prev_error_size = 1
        if self.hparams.get("use_prev_error", True):
            in_size = conv_out_size + 2 * state_size + action_size + prev_error_size
        else:
            in_size = conv_out_size + 2 * state_size + action_size
        use_drop_out = 'dropout_p' in self.hparams
        for hidden_size in self.hparams['fc_layer_sizes']:
            fc_layers.append(nn.Linear(in_size, hidden_size))
            fc_layers.append(nn.LeakyReLU())
            if use_drop_out:
                fc_layers.append(nn.Dropout(p=self.hparams.get('dropout_p', 0.0)))
            in_size = hidden_size

        self._final_hidden_dim = self.hparams['fc_layer_sizes'][-1]
        self.no_lstm = self.hparams.get('no_lstm', False)
        if not self.no_lstm:
            fc_layers.append(nn.LSTM(self._final_hidden_dim, self.hparams['rnn_size'], 1))

        self.conv_encoder = torch.nn.Sequential(*conv_layers)
        self.fc = torch.nn.Sequential(*fc_layers)

        if self.no_lstm:
            self.output_layer = nn.Linear(2 * self._final_hidden_dim, 1)
        else:
            self.output_layer = nn.Linear(self._final_hidden_dim, 1)

        self.debug = DebuggingViz(self.scenario, self.hparams.state_keys, self.hparams.action_keys)
        self.local_env_helper = LocalEnvHelper(h=self.local_env_h_rows, w=self.local_env_w_cols,
                                               c=self.local_env_c_channels,
                                               get_local_env_module=get_local_environment_torch)
        self.robot_info = RobotVoxelgridInfo(joint_positions_key=add_predicted_hack('joint_positions'))
        self.vg_info = VoxelgridInfo(h=self.local_env_h_rows,
                                     w=self.local_env_w_cols,
                                     c=self.local_env_c_channels,
                                     state_keys=self.point_state_keys_pred,
                                     jacobian_follower=self.scenario.robot.jacobian_follower,
                                     robot_info=self.robot_info,
                                     include_robot_geometry=self.include_robot_geometry,
                                     scenario=self.scenario,
                                     )

        self.has_checked_training_mode = False

        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.train_stat_scores = torchmetrics.classification.BinaryStatScores()
        self.val_stat_scores = torchmetrics.classification.BinaryStatScores()

    def _input_dict_to_conv_and_fc_layer(self, inputs):
        if self.local_env_helper.device != self.device:
            self.local_env_helper.to(self.device)
        local_env, local_origin_point = self.get_local_env(inputs)
        batch_size, time = inputs['time_idx'].shape[0:2]
        voxel_grids = self.vg_info.make_voxelgrid_inputs(inputs, local_env, local_origin_point, batch_size, time,
                                                         viz=debug_vgs())
        if debug_vgs():
            b = 0
            for t in range(voxel_grids.shape[1]):
                self.debug.plot_pred_state_rviz(inputs, b, t, 'pred_inputs')
                for i in range(voxel_grids.shape[2]):
                    raster_dict = {
                        'env':          voxel_grids[b, t, i].cpu().numpy(),
                        'res':          inputs['res'][b].cpu().numpy(),
                        'origin_point': local_origin_point[b].cpu().numpy(),
                    }

                    self.scenario.send_occupancy_tf(raster_dict, parent_frame_id='robot_root',
                                                    child_frame_id='local_env_vg')
                    raster_msg = environment_to_vg_msg(raster_dict, frame='local_env_vg', stamp=rospy.Time.now())
                    self.debug.raster_debug_pubs[i].publish(raster_msg)

        if self.hparams.get('use_sdf', False):
            import tensorflow as tf
            from moonshine.gpu_config import limit_gpu_mem
            from moonshine.tfa_sdf import build_sdf_3d
            try:
                limit_gpu_mem(None)
            except RuntimeError:
                pass

            res = tf.convert_to_tensor(inputs['res'].cpu().numpy())
            voxel_grids_tf = tf.convert_to_tensor(voxel_grids.cpu().numpy())
            sdf_tf = voxel_grids.clone()
            for t in range(2):
                if self.hparams.get('env_only_sdf', False):
                    # 0 is the channel for the environment
                    sdf_tf[:, t, 0] = torch.from_numpy(build_sdf_3d(voxel_grids_tf[:, t, 0], res).numpy())
                else:
                    for c in range(5):
                        sdf_tf[:, t, c] = torch.from_numpy(build_sdf_3d(voxel_grids_tf[:, t, c], res).numpy())
            voxel_grids = sdf_tf

        states = {k: inputs[add_predicted_hack(k)] for k in self.hparams.state_keys}
        states_local_frame = self.scenario.put_state_local_frame_torch(states)
        states_local_frame_list = list(states_local_frame.values())
        actions = {k: inputs[k] for k in self.hparams.action_keys}
        all_but_last_states = {k: v[:, :-1] for k, v in states.items()}
        actions = self.scenario.put_action_local_frame(all_but_last_states, actions)
        padded_actions = [F.pad(v, [0, 0, 0, 1, 0, 0]) for v in actions.values()]
        states_robot_frame = self.scenario.put_state_robot_frame(states)
        states_robot_frame_list = list(states_robot_frame.values())
        num_vg_layers = voxel_grids.shape[2]
        flat_voxel_grids = voxel_grids.reshape(
            [-1, num_vg_layers, self.local_env_h_rows, self.local_env_w_cols, self.local_env_c_channels])
        flat_conv_h = self.conv_encoder(flat_voxel_grids)
        conv_h = flat_conv_h.reshape(batch_size, time, -1)
        # NOTE: maybe we should be using the previous predicted error?
        prev_pred_error = inputs['error'][:, 0]
        prev_pred_error_time = prev_pred_error.unsqueeze(-1).unsqueeze(-1)
        padded_prev_pred_error = F.pad(prev_pred_error_time, [0, 0, 0, 1, 0, 0])
        if self.hparams.get("use_prev_error", True):
            cat_args = [conv_h,
                        padded_prev_pred_error] + states_robot_frame_list + states_local_frame_list + padded_actions
        else:
            cat_args = [conv_h] + states_robot_frame_list + states_local_frame_list + padded_actions
        fc_in = torch.cat(cat_args, -1)
        return batch_size, fc_in

    def forward(self, inputs: Dict[str, torch.Tensor]):
        if not self.has_checked_training_mode:
            self.has_checked_training_mode = True
            print(f"Training Mode? {self.training}")

        batch_size, fc_in = self._input_dict_to_conv_and_fc_layer(inputs)

        if self.no_lstm:
            fc_out_h = self.fc(fc_in)
            out_h = fc_out_h.reshape(batch_size, -1)
            predicted_errors = self.output_layer(out_h)  # [b, 1]
            predicted_error = predicted_errors.squeeze(1)  # [b]
        else:
            out_h, _ = self.fc(fc_in)
            # for every timestep's output, map down to a single scalar, the logit for accept probability
            predicted_errors = self.output_layer(out_h)  # [b, 1]
            predicted_error = predicted_errors[:, 1:].squeeze(1).squeeze(1)  # [b]

        return predicted_error

    def get_local_env(self, inputs):
        batch_size = inputs['time_idx'].shape[0]
        state_0 = {k: inputs[add_predicted_hack(k)][:, 0] for k in self.hparams.point_state_keys}

        local_env_center = self.scenario.local_environment_center_differentiable_torch(state_0)
        local_env, local_origin_point = self.local_env_helper.get(local_env_center, inputs, batch_size)

        return local_env, local_origin_point

    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs):
        true_error_after = inputs['error'][:, 1]
        mae = (outputs - true_error_after).abs().mean()
        mse_batch = F.mse_loss(outputs, true_error_after, reduction='none')
        mse = mse_batch.mean()
        true_error_after_binary = (true_error_after < self.hparams['error_threshold']).float()
        logits = -outputs
        biased_mse = (mse_batch * torch.exp(-10 * true_error_after)).mean()

        bias_mult = torch.exp(-1 * true_error_after)
        #bce = F.binary_cross_entropy_with_logits(logits, true_error_after_binary)
        asym_mse = self._c1 * torch.mean(bias_mult * torch.pow(F.relu(true_error_after - outputs), 2)) + self._c2 * torch.mean(bias_mult * torch.pow(F.relu(outputs - true_error_after), 2))

        if self.hparams.get("loss_type", None) == 'MAE':
            loss = mae
        elif self.hparams.get("loss_type", None) == 'BCE':
            loss = bce
        elif self.hparams.get("loss_type", None) == 'biased_mse':
            loss = biased_mse
        elif self.hparams.get("loss_type", None) == 'asym_biased_mse':
            loss = asym_mse
        else:
            raise ValueError

        return loss, mse, mae, mae

    def training_step(self, train_batch: Dict[str, torch.Tensor], batch_idx):
        outputs = self.forward(train_batch)
        loss, mse, mae, bce = self.compute_loss(train_batch, outputs)
        self.log('train_loss', loss)
        self.log('train_mae', mae)
        self.log('train_mse', mse)
        self.log('train_bce', bce)
        if batch_idx == 0:
            self.easy_accuracies = []
            self.easy_errors = []
            self.easy_pred_errors = []
            self.hard_accuracies = []
            self.hard_errors = []
            self.hard_pred_errors = []
        with torch.no_grad():
            pred_error = outputs
            pred_error_binary = (pred_error < self.hparams['error_threshold']).detach().cpu()
            true_error_after = train_batch['error'][:, 1]
            true_error_after_binary = (true_error_after < self.hparams['error_threshold']).int().detach().cpu()
            self.train_stat_scores(pred_error_binary, true_error_after_binary)

            # Check accuracy over "pours"
            if batch_idx == self.current_epoch % 50:
                for sample_i in range(train_batch["target_volume"].shape[0]):
                    pour_error, pour_type, sample_error = get_pour_type_and_error(train_batch, None, pred_error,
                                                                                  sample_i)
                    if pour_type == "easy_pour":
                        self.easy_errors.append(true_error_after[sample_i].detach().item())
                        self.easy_pred_errors.append(pred_error[sample_i].detach().item())
                    if pour_type == "hard_pour":
                        self.hard_errors.append(true_error_after[sample_i].detach().item())
                        self.hard_pred_errors.append(pred_error[sample_i].detach().item())
            self.log(f'train_easy_pour_pred_error', np.mean(self.easy_pred_errors))
            self.log(f'train_hard_pour_pred_error', np.mean(self.hard_pred_errors))
            self.log(f'train_easy_pour_error', np.mean(self.easy_errors))
            self.log(f'train_hard_pour_error', np.mean(self.hard_errors))
        return loss
    def on_train_epoch_end(self):
        tp, fp, tn, fn, _ = self.train_stat_scores.compute()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        self.log('train_fpr', fpr)
        self.log('train_tnr', 1 - fpr)
        self.log('train_tpr', tpr)
        self.log('train_fnr', 1 - tpr)
        # reset all metrics
        self.train_stat_scores.reset()

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx):
        pred_error = self.forward(val_batch)
        loss, mse, mae, bce = self.compute_loss(val_batch, pred_error)
        true_error_after = val_batch['error'][:, 1]
        signed_loss = pred_error - true_error_after
        true_error_after_binary = (true_error_after < self.hparams['error_threshold']).int()
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        self.log('val_mse', mse)
        self.log('val_bce', bce)
        if self.hparams.get("loss_type", 'mse') == 'BCE':
            logits = -pred_error
            pred_error_probabilities = torch.sigmoid(logits)
            self.val_accuracy(pred_error_probabilities, true_error_after_binary)
            self.val_stat_scores(pred_error_probabilities, true_error_after_binary)
        else:
            pred_error_binary = pred_error < self.hparams['error_threshold']
            self.val_accuracy(pred_error_binary, true_error_after_binary)
            self.val_stat_scores(pred_error_binary, true_error_after_binary)
        return loss

    def test_step(self, test_batch, batch_idx):
        pred_error = self.forward(test_batch)
        loss, mse, mae, bce = self.compute_loss(test_batch, pred_error)
        true_error_after = test_batch['error'][:, 1]
        signed_loss = pred_error - true_error_after
        true_error_after_binary = (true_error_after < self.hparams['error_threshold']).int()
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_mse', mse)
        self.log('test_bce', bce)
        if self.hparams.get("loss_type", 'mse') == 'BCE':
            logits = -pred_error
            pred_error_probabilities = torch.sigmoid(logits)
            self.test_accuracy(pred_error_probabilities, true_error_after_binary)
        else:
            pred_error_binary = pred_error < self.hparams['error_threshold']
            self.test_accuracy(pred_error_binary, true_error_after_binary)
        return loss

    def on_test_epoch_end(self):
        self.log('test_accuracy', self.test_accuracy.compute())

        # reset all metrics
        self.test_accuracy.reset()

    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute())  # logs the metric result/value
        tp, fp, tn, fn, _ = self.val_stat_scores.compute()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        self.log('val_fpr', fpr)
        self.log('val_tnr', 1 - fpr)
        self.log('val_tpr', tpr)
        self.log('val_fnr', 1 - tpr)

        # reset all metrics
        self.val_accuracy.reset()
        self.val_stat_scores.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.get('weight_decay', 0))


class MDEConstraintChecker:

    def __init__(self, checkpoint):
        self.model: MDE = load_model_artifact(checkpoint, MDE, project='mde', version='best', user='armlab')
        self.model.eval()
        self.model.cuda()
        self.model.local_env_helper.to("cuda")
        self.horizon = 2
        self.name = 'MDE'

    def check_constraint(self, environment: Dict, states_sequence: List[Dict], actions: List[Dict]):
        inputs = self.states_and_actions_to_torch_inputs(states_sequence, actions, environment)

        pred_error = remove_batch(self.model(add_batch(inputs)))
        return pred_error.detach().cpu().numpy()

    def states_and_actions_to_torch_inputs(self, states_sequence, actions, environment):
        states_dict = sequence_of_dicts_to_dict_of_tensors(states_sequence, device=self.model.device)
        actions_dict = sequence_of_dicts_to_dict_of_tensors(actions, device=self.model.device)
        inputs = {}
        environment = torchify(environment, device=self.model.device)
        inputs.update(environment)
        for action_key in self.model.hparams.action_keys:
            inputs[action_key] = actions_dict[action_key]
        for state_metadata_key in self.model.hparams.state_metadata_keys:
            inputs[state_metadata_key] = states_dict[state_metadata_key]
        for state_key in self.model.hparams.state_keys:
            planned_state_key = add_predicted(state_key)
            inputs[planned_state_key] = states_dict[state_key]
        if 'joint_names' in states_dict:
            inputs[add_predicted('joint_names')] = states_dict['joint_names']
        if 'joint_positions' in states_dict:
            inputs[add_predicted('joint_positions')] = states_dict['joint_positions']
        if 'error' in states_dict:
            inputs['error'] = states_dict['error'][:, 0]
        inputs['time_idx'] = torch.arange(2, dtype=torch.float32, device=self.model.device)
        return inputs


if __name__ == '__main__':
    rospy.init_node("mde_torch_test")
    config_path = pathlib.Path("gp_mde_configs/gp_mde_test.yaml")
    c = GPMDEConstraintChecker(config_path)
    import pickle

    with open("mde_test_inputs.pkl", 'rb') as f:
        env, states, actions = pickle.load(f)
    mde_outputs = c.check_constraint(env, states, actions)
    print(mde_outputs)

    # DEBUGGING
    # import pickle
    # with open("planning_first_transition_5.pkl", 'wb') as f:
    #     data = (environment, states_sequence, actions)
    #     pickle.dump(data, f)
