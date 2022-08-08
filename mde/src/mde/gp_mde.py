import os
from typing import Dict, List

import gpytorch
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from botorch.models import HeteroskedasticSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import nn

from link_bot_pycommon.load_wandb_model import load_model_artifact, get_gp_training_artifact
from moonshine.gpytorch_tools import mutate_dict_to_cpu, mutate_dict_to_cuda, TFStandardScaler
from moonshine.torch_and_tf_utils import add_batch
from moonshine.torch_datasets_utils import my_collate
from .gp_mde_base_classes import DeepExactGP
from .torch_mde import MDE, MDEConstraintChecker


class GPRMDE(MDE):
    def __init__(self, train_dataset=None, **hparams):
        MDE.__init__(self, **hparams)
        self.using_het_gp = False
        if "from_checkpoint" in self.hparams:
            self.hparams["gp_checkpoint"] = self.hparams["from_checkpoint"]
        self._gp_input_dim = self.hparams["gp_input_dim"]
        self.compressing_output_layer = nn.Linear(2 * self._final_hidden_dim, self._gp_input_dim)
        self._embedded_training_data = None
        self._loaded_all_train_data = False
        self._num_gp_training_points = self.hparams['num_gp_training_points']
        self.train_dataset = train_dataset
        self.beta = self.hparams['beta']
        self._grid_size = self.hparams["grid_size"]
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        self._train_batch = None
        self.is_heteroskedastic = self.hparams.get("is_heteroskedastic", False)
        if self.hparams.get("gp_checkpoint", None) is not None:
            self._load_existing_gp_data(self.hparams["gp_checkpoint"])
        self.val_p, self.val_n, self.val_tp, self.val_fp = torchmetrics.SumMetric(), torchmetrics.SumMetric(), torchmetrics.SumMetric(), torchmetrics.SumMetric()
        self.test_p, self.test_n, self.test_tp, self.test_fp = torchmetrics.SumMetric(), torchmetrics.SumMetric(), torchmetrics.SumMetric(), torchmetrics.SumMetric()
        self.val_sum_metrics = [self.val_p, self.val_n, self.val_tp, self.val_fp]
        self.test_sum_metrics = [self.test_p, self.test_n, self.test_tp, self.test_fp]
        self._setup_for_training()

    def _setup_collated_random_training_batch(self):
        possible_data_idxs = np.arange(len(self.train_dataset))  # TODO lagrassa make random?
        data_idxs = np.random.choice(possible_data_idxs, replace=False, size=(self._num_gp_training_points,))
        training_points_before_collate = []
        for idx, i in enumerate(data_idxs):
            training_points_before_collate.append(self.train_dataset[i])
        self._train_batch = my_collate(training_points_before_collate)

    def _setup_label_scaler(self, representative_labels=None):
        if representative_labels is None:
            representative_labels = []
            for i in range(len(self.train_dataset)):
                representative_labels.append(self.train_dataset[i]["error"][1].item())
        tf_representative_labels = torch.Tensor(representative_labels)
        self.error_scaler = TFStandardScaler().fit(tf_representative_labels)

    def make_gp(self):
        # Train label scaler and add training points to GP
        error_labels = self.error_scaler.transform(self._train_error_labels).cuda()
        # self.model = DeepExactGPDetachedTrainingData(self._train_batch, error_labels, self.likelihood, self.input_data_to_gp_input, self._gp_input_dim, self._grid_size, self.features_to_gp_input, self.input_data_to_features)
        self.model_homoskedastic = DeepExactGP(self._train_batch, error_labels, self.likelihood,
                                               self.input_data_to_gp_input)
        self.model_homoskedastic.mean_module = gpytorch.means.ConstantMean().cuda()
        self.model_homoskedastic.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=self._gp_input_dim)).cuda()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_homoskedastic.likelihood,
                                                            self.model_homoskedastic)
        self.model = self.model_homoskedastic

    def make_heteroskedastic_gp(self):
        error_labels = self.error_scaler.transform(self._train_error_labels).cuda()
        self.model_homoskedastic.eval()
        observed_var = torch.pow(self.model_homoskedastic(
            self.model_homoskedastic.train_inputs).mean - self.model_homoskedastic.train_targets, 2)
        gp_input = self.input_data_to_gp_input(self.model_homoskedastic.train_inputs)
        self.model = HeteroskedasticSingleTaskGP(gp_input, error_labels.reshape(-1, 1), observed_var.reshape(-1, 1))
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.using_het_gp = True

    def _load_existing_gp_data(self, gp_checkpoint):
        artifact = get_gp_training_artifact(gp_checkpoint, project='gpmde', version='v0', user='lagrassa')
        artifact_dir = artifact.download()
        self._train_batch = np.load(os.path.join(artifact_dir, "train_inputs.npy"), allow_pickle=True).item()
        mutate_dict_to_cpu(self._train_batch)
        self._train_error_labels = self._train_batch["error"][:, 1]
        self._setup_label_scaler(representative_labels=self._train_error_labels)

    def _setup_for_training(self):
        if self._train_batch is None:
            self._setup_collated_random_training_batch()
            self._train_error_labels = self._train_batch["error"][:, 1]
            self._setup_label_scaler(representative_labels=self._train_error_labels)
        self._train_error_labels = self._train_batch["error"][:, 1].clone().cuda()

        self.make_gp()
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        self.train_d_hats = []
        self.val_d_hats, self.val_d_gts, self.val_stds = [], [], []
        self.test_d_hats, self.test_d_gts, self.test_stds = [], [], []

    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs):
        # error_after = torch.from_numpy(self._error_forward_tf(inputs["error"][:, 1]).astype(np.float32)).cuda()
        true_error_after = self._error_forward_tf(inputs["error"][:, 1])
        loss = -self.mll(outputs, true_error_after)
        wandb.log({'mll_train_loss': loss.item()})
        mse_batch = self.mse(outputs.mean, true_error_after)
        wandb.log({'train_mse_grad': mse_batch.item()})
        # return loss + 0.1*biased_mse #+ 0.1*mae_loss
        return loss  # + 0.1*mse_batch #+ 0.1*mae_loss

    def change_gp_training_points(self, train_batch, error_label, replace=True):
        with torch.no_grad():
            projected_x, batch_size = self.input_data_to_embedding(train_batch)
        if self._embedded_training_data is None:
            self._embedded_training_data = projected_x.reshape(batch_size, -1)
            self._ys = error_label
        else:
            self._embedded_training_data = torch.cat((self._embedded_training_data, projected_x), dim=0)
            self._ys = torch.cat((self._ys, error_label), dim=0)
        self.model.set_train_data(self._embedded_training_data, self._ys, strict=False)

    def _error_inv_tf(self, error_batch_tensor):
        return self.error_scaler.inverse_transform(error_batch_tensor)

    def _error_forward_tf(self, error_batch_tensor):
        return self.error_scaler.transform(error_batch_tensor)

    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs, posterior_mode=False):
        # error_after = torch.from_numpy(self._error_forward_tf(inputs["error"][:, 1]).astype(np.float32)).cuda()
        true_error_after = self._error_forward_tf(inputs["error"][:, 1])
        if posterior_mode:
            # penalty = torch.exp(-10*inputs["error"][:,1])
            diff = true_error_after - outputs.mean
            # biased_diff = penalty * diff
            mse_batch = torch.pow(diff, 2).mean()
            # mae_batch = self.mae(outputs.mean, true_error_after)
            wandb.log({'train_mse_post': mse_batch.item()})
            # wandb.log({'biased_train_mse': biased_mse_batch.item()})
            loss = 100 * mse_batch
        else:
            mll_loss = -self.mll(outputs, true_error_after)
            wandb.log({'mll_train_loss': mll_loss.item()})
            loss = mll_loss
        return loss

    def training_step(self, train_batch: Dict[str, torch.Tensor], batch_idx):
        if self.using_het_gp:
            error_labels = self.error_scaler.transform(self._train_error_labels).cuda()
            observed_var = torch.pow(self.model_homoskedastic(self._train_batch).mean - error_labels, 2)
            self.model.set_train_data(self.input_data_to_gp_input(self._train_batch), error_labels)
            outputs = self.model(self.input_data_to_gp_input(train_batch))
        else:
            outputs = self.model(train_batch)
        loss = self.compute_loss(train_batch, outputs)
        wandb.log({"lengthscale": self.model.covar_module.base_kernel.lengthscale})
        wandb.log({"noise": self.model.likelihood.noise})
        self.log('train_loss', loss.item())
        return loss

    def _eval_model_on_data(self, x, switch_to_train=True):
        self.model.eval()
        self.mll.eval()
        self.scale_to_bounds.eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            outputs = self.model(x)
        if switch_to_train:
            self.model.train()
            self.mll.train()
        return outputs

    def _get_unscaled_mean_and_std(self, dhat_batch):
        error_unscaled = self._error_inv_tf(dhat_batch.mean)
        std_scaled = torch.sqrt(dhat_batch.variance)
        std_unscaled = (std_scaled * self.error_scaler.std).flatten()
        return error_unscaled, std_unscaled

    def training_epoch_end(self, inp):
        # make a table
        dhat_batch = self._eval_model_on_data(self._train_batch)
        error_unscaled, std_unscaled = self._get_unscaled_mean_and_std(dhat_batch)
        train_mse = self.mse(error_unscaled, self._train_error_labels)
        train_d_hats = error_unscaled
        self.log('train_mse', train_mse)
        std_scaled = torch.sqrt(dhat_batch.variance)
        # print("Train variance", pred_stds)
        self.log_dgt_dhat_table_and_plot(dhat_batch.mean.cpu().detach().numpy().flatten(), self._error_forward_tf(
            self._train_error_labels).cpu().detach().numpy().flatten(), std_scaled.cpu().detach().numpy().flatten(),
                                         label="train_scaled_space")
        self.log_dgt_dhat_table_and_plot(train_d_hats.flatten(),
                                         self._train_error_labels.cpu().detach().numpy().flatten(), std_unscaled,
                                         label="train")
        self.train_d_hats = []

    def log_dgt_dhat_table_and_plot(self, d_hats, d_gts, pred_stds, label="default"):
        datas = [[dgt, dhat, pred_std] for dgt, dhat, pred_std in zip(d_gts, d_hats, pred_stds)]
        table = wandb.Table(data=datas, columns=[f"{label}_d_gt", f"{label}_d_hat", f"{label}_std"])
        wandb.log({f"ddgt_{label}": wandb.plot.scatter(table, f"{label}_d_gt", f"{label}_d_hat")})

    def test_step(self, test_batch, batch_idx):
        true_error = test_batch['error'][:, 1]
        pred_error = self._eval_model_on_data(test_batch)
        std_scaled = torch.sqrt(pred_error.variance)
        std = std_scaled * self.error_scaler.std
        # print("Val variance", std)
        # print("Val mean", pred_error.mean)
        try:
            loss = self.compute_loss(test_batch, self.model(test_batch))
            self.log('test_loss', loss)
        except:
            print("Numerical errors with computing val loss")
            dummy_val = 1.
            self.log('test_loss', dummy_val)
            loss = torch.Tensor([dummy_val])
        # mean_pred_error_unscaled = torch.from_numpy(self._error_inv_tf(pred_error.mean)).cuda()
        mean_pred_error_unscaled = self._error_inv_tf(pred_error.mean)
        true_error_thresholded = true_error < self.hparams.error_threshold
        pred_error_thresholded = mean_pred_error_unscaled + self.beta * std < self.hparams.error_threshold
        signed_loss = mean_pred_error_unscaled - true_error
        self.test_accuracy(pred_error_thresholded, true_error_thresholded)  # updates the metric
        self.test_p(torch.sum(true_error_thresholded))
        self.test_tp(torch.sum(torch.logical_and(true_error_thresholded, pred_error_thresholded)))
        self.test_fp(torch.sum(torch.logical_and(pred_error_thresholded, torch.logical_not(true_error_thresholded))))
        self.test_n(torch.sum(torch.logical_not(true_error_thresholded)))
        self.log('pred_minus_true_error', signed_loss)
        # Bookkeeping
        self.test_d_hats.extend(mean_pred_error_unscaled.detach().cpu().numpy().flatten())
        self.test_d_gts.extend(true_error.detach().cpu().numpy().flatten())
        self.test_stds.extend(std.detach().cpu().numpy().flatten())
        return loss

    def on_test_epoch_end(self):
        wandb.log({"test_mae": np.mean(np.abs(np.array(self.test_d_hats) - np.array(self.test_d_gts)))})
        if len(self.test_d_hats):
            self.log('test_accuracy', self.test_accuracy.compute().item())  # logs the metric result/testue
            wandb.log({'test_accuracy': self.test_accuracy.compute().item()})
            num_p = self.test_p.compute().item()
            num_n = self.test_n.compute().item()
            num_fp = self.test_fp.compute().item()
            num_tp = self.test_tp.compute().item()
            wandb.log({'true_positive_rate': num_tp / num_p})
            self.log('true_positive_rate', num_tp / num_p)
            wandb.log({'false_positive_rate': num_fp / num_n})
            self.log('false_positive_rate', num_fp / num_n)
        # reset all metrics
        self.log_dgt_dhat_table_and_plot(self.test_d_hats, self.test_d_gts, self.test_stds, label="test")
        self.test_d_hats = []
        self.test_d_gts = []
        self.test_stds = []

        # reset all metrics
        self.test_accuracy.reset()
        for sum_metric in self.test_sum_metrics:
            sum_metric.reset()

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx):
        true_error = val_batch['error'][:, 1]
        dhat_batch = self._eval_model_on_data(val_batch)
        std_scaled = torch.sqrt(dhat_batch.variance)
        # print("Val variance", std)
        # print("Val mean", pred_error.mean)
        try:
            loss = self.compute_loss(val_batch, self.model(val_batch))
            self.log('val_loss', loss)
        except:
            print("Numerical errors with computing val loss")
            dummy_val = 1.
            self.log('val_loss', dummy_val)
            loss = torch.Tensor([dummy_val])
        # mean_pred_error_unscaled = torch.from_numpy(self._error_inv_tf(pred_error.mean)).cuda()
        mean_pred_error_unscaled = self._error_inv_tf(dhat_batch.mean)
        error_unscaled, std_unscaled = self._get_unscaled_mean_and_std(dhat_batch)
        true_error_thresholded = true_error < self.hparams.error_threshold
        pred_error_thresholded = error_unscaled + self.beta * std_unscaled < self.hparams.error_threshold
        signed_loss = error_unscaled - true_error
        self.val_accuracy(pred_error_thresholded, true_error_thresholded)  # updates the metric
        self.val_p(torch.sum(true_error_thresholded))
        self.val_tp(torch.sum(torch.logical_and(true_error_thresholded, pred_error_thresholded)))
        self.val_fp(torch.sum(torch.logical_and(pred_error_thresholded, torch.logical_not(true_error_thresholded))))
        self.val_n(torch.sum(torch.logical_not(true_error_thresholded)))
        self.log('pred_minus_true_error', signed_loss.abs().mean())
        # Bookkeeping
        self.val_d_hats.extend(mean_pred_error_unscaled.detach().cpu().numpy().flatten())
        self.val_d_gts.extend(true_error.detach().cpu().numpy().flatten())
        self.val_stds.extend(std_unscaled.detach().cpu().numpy().flatten())
        return loss

    def validation_epoch_end(self, _):
        wandb.log({"val_mae": np.mean(np.abs(np.array(self.val_d_hats) - np.array(self.val_d_gts)))})
        if len(self.val_d_hats):
            self.log('val_accuracy', self.val_accuracy.compute().item())  # logs the metric result/value
            wandb.log({'val_accuracy': self.val_accuracy.compute().item()})
            num_p = self.val_p.compute().item()
            num_n = self.val_n.compute().item()
            num_fp = self.val_fp.compute().item()
            num_tp = self.val_tp.compute().item()
            if num_p > 0:
                wandb.log({'true positive rate': num_tp / num_p})
            if num_n > 0:
                wandb.log({'false positive rate': num_fp / num_n})
        print("Val accuracy", self.val_accuracy.compute())
        # reset all metrics
        self.log_dgt_dhat_table_and_plot(self.val_d_hats, self.val_d_gts, self.val_stds, label="val")
        self.val_d_hats = []
        self.val_d_gts = []
        self.val_stds = []
        self.val_accuracy.reset()
        for sum_metric in self.val_sum_metrics:
            sum_metric.reset()

    def input_data_to_embedding(self, input_data):
        batch_size, fc_in = self._input_dict_to_conv_and_fc_layer(input_data)
        if self.no_lstm:
            fc = self.fc(fc_in)
        else:
            fc, _ = self.fc(fc_in)
        out_h = fc.reshape(batch_size, -1)
        return out_h, batch_size

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        pl.LightningModule.load_state_dict(self, state_dict, strict=False)

    def features_to_gp_input(self, out_h):
        out_h = nn.LeakyReLU()(out_h)
        out_h = self.compressing_output_layer(out_h)
        projected_x = self.scale_to_bounds(out_h)  # Make the NN values "nice"
        return projected_x

    def input_data_to_gp_input(self, input_data):
        # mutate_dict_to_cuda(input_data)
        out_h, batch_size = self.input_data_to_embedding(input_data)
        projected_x = self.features_to_gp_input(out_h)
        return projected_x

    def input_data_to_features(self, input_data):
        # mutate_dict_to_cuda(input_data)
        out_h, _ = self.input_data_to_embedding(input_data)
        return out_h

    def forward(self, input_data):
        projected_x = self.input_data_to_gp_input(input_data)
        mean_x = self.model.mean_module(projected_x)
        covar_x = self.model.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def eval(self):
        self.model.eval()
        self.mll.eval()

    def predict(self, test_x):
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._model_heter.posterior(test_x, observation_noise=True)
        return pred


class GPMDEConstraintChecker(MDEConstraintChecker):
    def __init__(self, checkpoint):
        self.model = load_model_artifact(checkpoint, GPRMDE, project='gpmde', version='best', user='lagrassa')
        mutate_dict_to_cuda(self.model.model.train_inputs)
        self.model = self.model.to("cuda")
        self.model.eval()
        self.horizon = 2
        self.name = 'MDE'

    def check_constraint(self, environment: Dict, states_sequence: List[Dict], actions: List[Dict]):
        inputs = self.states_and_actions_to_torch_inputs(states_sequence, actions, environment)
        batch_inputs = add_batch(inputs)
        mutate_dict_to_cuda(batch_inputs)
        dhat_batch = self.model._eval_model_on_data(batch_inputs, switch_to_train=False)
        error_unscaled, std_unscaled = self.model._get_unscaled_mean_and_std(dhat_batch)
        return (error_unscaled[0].cpu().numpy() + 0 * std_unscaled).item()
