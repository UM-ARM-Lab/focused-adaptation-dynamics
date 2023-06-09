#!/usr/bin/env python

import pathlib
from datetime import datetime
from typing import Optional, Union, List

import git
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from wandb.util import generate_id

from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from link_bot_data.visualization import init_viz_env, viz_pred_actual_t
from link_bot_data.wandb_datasets import get_dataset_with_version
from link_bot_pycommon.load_wandb_model import load_model_artifact, model_artifact_path
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.my_pl_callbacks import HeartbeatCallback
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torch_datasets_utils import dataset_skip
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.torch_udnn import UDNN
from state_space_dynamics.udnn_data_module import UDNNDataModule

PROJECT = 'udnn'


def load_udnn_model_wrapper(checkpoint, with_joint_positions=False):
    model = load_model_artifact(checkpoint, UDNN, 'udnn', version='best', user='armlab',
                                with_joint_positions=with_joint_positions)
    model.eval()
    return model


def fine_tune_main(dataset_dir: Union[pathlib.Path, List[pathlib.Path]],
                   checkpoint: str,
                   params_filename: pathlib.Path,
                   batch_size: int,
                   epochs: int,
                   seed: int,
                   user: str,
                   steps: int = -1,
                   nickname: Optional[str] = None,
                   no_val: Optional[bool] = False,
                   take: Optional[int] = None,
                   skip: Optional[int] = None,
                   repeat: Optional[int] = None,
                   project=PROJECT,
                   **kwargs):
    pl.seed_everything(seed, workers=True)

    run_id = generate_id(length=5)
    if nickname is not None:
        run_id = nickname + '-' + run_id

    params = load_hjson(params_filename)
    params.update(kwargs)

    data_module = UDNNDataModule(dataset_dir,
                                 batch_size=batch_size,
                                 take=take,
                                 skip=skip,
                                 repeat=repeat,
                                 train_mode=params['train_mode'],
                                 val_mode=params['val_mode'],
                                 )
    data_module.add_dataset_params(params)
    params['repeat'] = repeat

    model = load_model_artifact(checkpoint, UDNN, project=project, version='latest', user=user, **params)

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', entity=user)
    if no_val:
        ckpt_cb = pl.callbacks.ModelCheckpoint(save_last=True, filename='{epoch:02d}', save_on_train_epoch_end=True)
    else:
        ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    hearbeat_callback = HeartbeatCallback(model.scenario)

    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=int(steps / batch_size) if steps != -1 else steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=999 if no_val else 1,
                         callbacks=[ckpt_cb, hearbeat_callback],
                         default_root_dir='wandb',
                         detect_anomaly=True,
                         gradient_clip_val=0.01)
    trainer.fit(model, data_module)
    wandb.finish()
    # eval_main(dataset_dir, run_id, mode='test', user=user, batch_size=batch_size)
    return run_id


def train_main(dataset_dir: pathlib.Path,
               params_filename: pathlib.Path,
               batch_size: int,
               epochs: int,
               seed: int,
               user: str,
               steps: int = -1,
               nickname: Optional[str] = None,
               no_val: Optional[bool] = False,
               checkpoint: Optional = None,
               take: Optional[int] = None,
               skip: Optional[int] = None,
               repeat: Optional[int] = None,
               project=PROJECT,
               **kwargs):
    pl.seed_everything(seed, workers=True)

    params = load_hjson(params_filename)

    data_module = UDNNDataModule(dataset_dir,
                                 batch_size=batch_size,
                                 take=take,
                                 skip=skip,
                                 repeat=repeat,
                                 train_mode=params['train_mode'],
                                 val_mode=params['val_mode'])
    data_module.add_dataset_params(params)

    # add some extra useful info here
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    params['repeat'] = repeat
    params['sha'] = sha
    params['start-train-time'] = stamp
    params['batch_size'] = batch_size
    params['seed'] = seed
    params['max_epochs'] = epochs
    params['steps'] = steps
    params['checkpoint'] = checkpoint

    wandb_kargs = {'entity': user}
    if checkpoint is None:
        ckpt_path = None
        run_id = generate_id(length=5)
        if nickname is not None:
            run_id = nickname + '-' + run_id
    else:
        ckpt_path = model_artifact_path(checkpoint, project, version='latest', user=user)
        run_id = checkpoint
        wandb_kargs['resume'] = True

    model = UDNN(**params)
    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)
    if no_val:
        ckpt_cb = pl.callbacks.ModelCheckpoint(save_last=True, filename='{epoch:02d}', save_on_train_epoch_end=True)
    else:
        ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    hearbeat_callback = HeartbeatCallback(model.scenario)
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=int(steps / batch_size) if steps != -1 else steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=999 if no_val else 1,
                         callbacks=[ckpt_cb, hearbeat_callback],
                         default_root_dir='wandb',
                         gradient_clip_val=0.01)
    wb_logger.watch(model)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
    wandb.finish()
    eval_main(dataset_dir, run_id, mode='test', user=user, batch_size=batch_size)
    return run_id


def eval_main(dataset_dir: pathlib.Path,
              checkpoint: str,
              mode: str,
              batch_size: int,
              user: str,
              take: Optional[int] = None,
              skip: Optional[int] = None,
              project=PROJECT,
              **kwargs):
    model = load_udnn_model_wrapper(checkpoint)
    model.eval()

    run_id = f'eval-{generate_id(length=5)}'
    eval_config = {
        'training_dataset':       model.hparams.dataset_dir,
        'eval_dataset':           dataset_dir.as_posix(),
        'eval_dataset_versioned': get_dataset_with_version(dataset_dir, PROJECT),
        'eval_checkpoint':        checkpoint,
        'eval_mode':              mode,
    }

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, tags=['eval'], config=eval_config, entity=user)
    trainer = pl.Trainer(gpus=1, enable_model_summary=False, logger=wb_logger)

    data_module = UDNNDataModule(dataset_dir, batch_size=batch_size, take=take, skip=skip)

    metrics = trainer.test(model, data_module, verbose=False)
    wandb.finish()

    print(f'run_id: {run_id}')
    for metrics_i in metrics:
        for k, v in metrics_i.items():
            print(f"{k:20s}: {v:0.6f}")

    import torch
    test_errors = torch.cat(model.test_errors, 0).reshape([-1])
    print(f"std test error: {torch.std(test_errors).detach().cpu().numpy():.2f}")

    return metrics


def viz_main(dataset_dir: pathlib.Path,
             checkpoint,
             mode: str,
             skip: Optional[int] = None,
             **kwargs):
    dataset_dir = fetch_udnn_dataset(dataset_dir)
    original_dataset = TorchDynamicsDataset(dataset_dir, mode)

    dataset = dataset_skip(original_dataset, skip)

    model = load_udnn_model_wrapper(checkpoint)

    s = original_dataset.get_scenario()

    dataset_anim = RvizAnimationController(n_time_steps=len(dataset), ns='trajs')

    n_examples_visualized = 0
    while not dataset_anim.done:
        inputs = dataset[dataset_anim.t()]
        print(inputs['example_idx'])

        if 'time_mask' in inputs:
            n_time_steps = int(inputs['time_mask'].sum())
        else:
            n_time_steps = inputs['time_idx'].shape[0]
        time_anim = RvizAnimationController(n_time_steps=n_time_steps)

        inputs_batch = torchify(add_batch(inputs))
        outputs_batch = model(inputs_batch)
        # low_error_mask = numpify(remove_batch(model.low_error_mask(inputs_batch, outputs_batch, global_step=100)))
        outputs = remove_batch(outputs_batch)

        time_anim.reset()
        while not time_anim.done:
            t = time_anim.t()

            init_viz_env(s, inputs, t)
            viz_pred_actual_t(original_dataset, model, inputs, outputs, s, t, threshold=0.08)
            # s.plot_weight_rviz(low_error_mask[t])
            time_anim.step()

        n_examples_visualized += 1

        dataset_anim.step()

    print(f"{n_examples_visualized:=}")
