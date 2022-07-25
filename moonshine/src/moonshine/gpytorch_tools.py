import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from sklearn.preprocessing import StandardScaler as SKStandardScaler


def custom_combine_batches(batches):
    ref_batch = batches[0]
    new_batch = {}
    for batch in batches:
        batch["env"] = _fix_env2(batch["env"])
    for key_name in ref_batch.keys():
        if isinstance(batches[0][key_name], torch.Tensor):
            new_batch[key_name] = torch.cat([batch[key_name] for batch in batches if key_name in batch.keys()], dim=0)
        if isinstance(batches[0][key_name], np.ndarray):
            new_batch[key_name] = np.vstack([batch[key_name] for batch in batches if key_name in batch.keys()])
    return new_batch


def mutate_dict_to_cpu(batch):
    for key_name in batch.keys():
        if isinstance(batch[key_name], torch.Tensor):
            batch[key_name] = batch[key_name].cpu()


def mutate_dict_to_cuda(batch):
    for key_name in batch.keys():
        if isinstance(batch[key_name], torch.Tensor) and batch[key_name].device.type == "cpu":
            batch[key_name] = batch[key_name].cuda()


def add_wandb_file_to_artifact(experiment, file_name, file_data, artifact):
    full_file_path = os.path.join(experiment.dir, file_name)
    np.save(full_file_path, file_data)
    artifact.add_file(full_file_path)


class TrainingDataSaveCB(pl.callbacks.Callback):
    def __init__(self, gpr_model):
        self.model = gpr_model

    def on_sanity_check_start(self, trainer, pl_module):
        train_inputs = self.model.model.train_inputs
        train_targets = self.model.model.train_targets
        error_scaler = self.model.error_scaler
        artifact_name = f"{trainer.logger.experiment.name}_training_data"
        artifact = wandb.Artifact(artifact_name, type="dataset")
        add_wandb_file_to_artifact(trainer.logger.experiment, "train_inputs.npy", self.model.model.train_inputs,
                                   artifact)
        add_wandb_file_to_artifact(trainer.logger.experiment, "train_targets.npy", self.model.model.train_targets.cpu(),
                                   artifact)
        add_wandb_file_to_artifact(trainer.logger.experiment, "error_scaler.npy", self.model.error_scaler, artifact)
        trainer.logger.experiment.log_artifact(artifact)


def _fix_env1(voxel_grid):
    old_device = voxel_grid.device
    if voxel_grid.shape[1:] != (60, 50, 60):
        voxel_grid = voxel_grid.cpu()
        voxel_grid = voxel_grid[:, 5:-5, :, :]
        voxel_grid = np.concatenate([np.zeros_like(voxel_grid[:, :, 0:5, :]), voxel_grid[:, :, :-2, :]], axis=2)
        voxel_grid = voxel_grid[:, :, :, 2:-5]
        assert voxel_grid[0].shape == (60, 50, 60)
        voxel_grid = torch.from_numpy(voxel_grid).to(old_device)
    return voxel_grid


def _fix_env2(voxel_grid):
    if voxel_grid.shape[1:] == (70, 50, 67):
        voxel_grid = voxel_grid[:, :, 1:-2, :]
        assert voxel_grid[0].shape == (70, 47, 67)
    return voxel_grid

class TFStandardScaler(SKStandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        self.mean = torch.mean(X)
        self.std = torch.std(X)
        return self

    def inverse_transform(self, X, copy=None):
        return (X*self.std) + self.mean

    def transform(self, X, copy=None):
        return (1./self.std)*(X - self.mean)
