#!/usr/bin/env python

import os
import pathlib
from datetime import datetime
from typing import Optional, Union, List

import git
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from wandb.util import generate_id

from link_bot_data.new_dataset_utils import fetch_mde_dataset
from link_bot_data.visualization import init_viz_env
from link_bot_data.wandb_datasets import get_dataset_with_version
from link_bot_pycommon.load_wandb_model import load_model_artifact, model_artifact_path
from mde.mde_data_module import MDEDataModule
from mde.torch_mde import MDE
from mde.torch_mde_dataset import TorchMDEDataset
from merrrt_visualization.rviz_animation_controller import RvizAnimationController
from moonshine.filepath_tools import load_hjson
from moonshine.gpytorch_tools import TrainingDataSaveCB
from moonshine.my_pl_callbacks import HeartbeatCallback
from moonshine.torch_and_tf_utils import add_batch, remove_batch
from moonshine.torch_datasets_utils import dataset_skip
from moonshine.torchify import torchify

PROJECT = 'mde'


def fine_tune_main(dataset_dir: Union[pathlib.Path, List[pathlib.Path]],
                   checkpoint: str,
                   params_filename: pathlib.Path,
                   batch_size: int,
                   epochs: int,
                   seed: int,
                   user: str,
                   steps: int = -1,
                   dryrun: bool = False,
                   is_nn_mde: Optional[bool] = True,
                   nickname: Optional[str] = None,
                   take: Optional[int] = None,
                   skip: Optional[int] = None,
                   repeat: Optional[int] = None,
                   train_mode='train',
                   val_mode='val',
                   no_val: Optional[bool] = False,
                   project=PROJECT,
                   **kwargs):
    pl.seed_everything(seed, workers=True)
    if dryrun:
        print("OFFLINE MODE")
        os.environ['WANDB_MODE'] = "offline"

    run_id = generate_id(length=5)
    if nickname is not None:
        run_id = nickname + '-' + run_id

    params = load_hjson(params_filename)
    params.update(kwargs)

    data_module = MDEDataModule(dataset_dir,
                                batch_size=batch_size,
                                take=take,
                                skip=skip,
                                repeat=repeat,
                                train_mode=train_mode,
                                val_mode=val_mode)
    data_module.add_dataset_params(params)
    data_module.setup()

    callbacks = []
    if is_nn_mde:
        model = load_model_artifact(checkpoint, MDE, project, version='best', user=user, **params)
    else:
        from mde.gp_mde import GPRMDE
        model = load_model_artifact(checkpoint, GPRMDE, project, version='best', user=user, gp_checkpoint=checkpoint, **params)
        callbacks.append(TrainingDataSaveCB(model))

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', entity=user)
    if no_val:
        ckpt_cb = pl.callbacks.ModelCheckpoint(save_last=True, filename='{epoch:02d}', save_on_train_epoch_end=True)
    else:
        ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    hearbeat_callback = HeartbeatCallback(model.scenario)
    callbacks.extend([ckpt_cb, hearbeat_callback])
    max_steps = max(1, int(steps / batch_size)) if steps != -1 else steps
    print(f"{max_steps=}")
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         max_epochs=epochs,
                         max_steps=max_steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=999 if no_val else 1,
                         callbacks=callbacks,
                         default_root_dir='wandb',
                         detect_anomaly=True,
                         )
    wb_logger.watch(model)
    trainer.fit(model, data_module)
    wandb.finish()

    return run_id


def train_main(dataset_dir: Union[pathlib.Path, List[pathlib.Path]],
               params_filename: pathlib.Path,
               batch_size: int,
               epochs: int,
               seed: int,
               user: str,
               steps: int = -1,
               dryrun: bool = False,
               is_nn_mde: Optional[bool] = True,
               nickname: Optional[str] = None,
               checkpoint: Optional = None,
               take: Optional[int] = None,
               skip: Optional[int] = None,
               repeat: Optional[int] = None,
               train_mode='train',
               val_mode='val',
               no_val: Optional[bool] = False,
               project=PROJECT,
               **kwargs):
    pl.seed_everything(seed, workers=True)
    if dryrun:
        print("OFFLINE MODE")
        os.environ['WANDB_MODE'] = "offline"

    params = load_hjson(params_filename)
    params.update(kwargs)

    data_module = MDEDataModule(dataset_dir,
                                batch_size=batch_size,
                                take=take,
                                skip=skip,
                                repeat=repeat,
                                train_mode=train_mode,
                                val_mode=val_mode)
    data_module.add_dataset_params(params)
    data_module.setup()

    # add some extra useful info here
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    params['sha'] = sha
    params['start-train-time'] = stamp
    params['batch_size'] = batch_size
    params['seed'] = seed
    params['epochs'] = epochs
    params['steps'] = steps
    params['checkpoint'] = checkpoint

    if checkpoint is None:
        ckpt_path = None
        run_id = generate_id(length=5)
        if nickname is not None:
            run_id = nickname + '-' + run_id
        wandb_kargs = {'entity': user}
    else:
        ckpt_path = model_artifact_path(checkpoint, project, version='latest', user=user)
        run_id = checkpoint
        wandb_kargs = {
            'entity': user,
            'resume': True,
        }
    callbacks = []
    if is_nn_mde:
        model = MDE(**params)
    else:
        from mde.gp_mde import GPRMDE
        model = GPRMDE(data_module.train_dataset, **params)
        callbacks.append(TrainingDataSaveCB(model))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"# params: {num_params}")

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, log_model='all', **wandb_kargs)
    if no_val:
        ckpt_cb = pl.callbacks.ModelCheckpoint(save_last=True, filename='{epoch:02d}', save_on_train_epoch_end=True)
    else:
        ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, filename='{epoch:02d}')
    hearbeat_callback = HeartbeatCallback(model.scenario)
    callbacks.extend([ckpt_cb, hearbeat_callback])
    max_steps = max(1, int(steps / batch_size)) if steps != -1 else steps
    print(f"{max_steps=}")
    trainer = pl.Trainer(gpus=1,
                         logger=wb_logger,
                         enable_model_summary=False,
                         detect_anomaly=True,
                         max_epochs=epochs,
                         max_steps=max_steps,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=999 if no_val else 1,
                         callbacks=callbacks,
                         default_root_dir='wandb')
    wb_logger.watch(model)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
    wandb.finish()

    # script = model.to_torchscript()
    # torch.jit.save(script, "model.pt")

    # eval_main(dataset_dir,
    #           run_id,
    #           mode='test',
    #           user=user,
    #           batch_size=batch_size)

    return run_id


def eval_main(dataset_dir: pathlib.Path,
              checkpoint: str,
              mode: str,
              batch_size: int,
              user: str,
              beta: Optional[float] = 2,
              is_nn_mde: Optional[int] = 1,
              take: Optional[int] = None,
              skip: Optional[int] = None,
              dryrun: Optional[bool] = False,
              project=PROJECT,
              **kwargs):
    if dryrun:
        print("OFFLINE MODE")
        os.environ['WANDB_MODE'] = "offline"

    if is_nn_mde:
        model = load_model_artifact(checkpoint, MDE, project, version='best', user=user)
    else:
        from mde.gp_mde import GPRMDE
        model = load_model_artifact(checkpoint, GPRMDE, project, version='best', user=user, gp_checkpoint=checkpoint)
        model.beta = beta

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())

    run_id = f'eval-{generate_id(length=5)}'
    eval_config = {
        'num_params':             num_params,
        'training_dataset':       model.hparams.dataset_dir,
        'eval_dataset':           dataset_dir.as_posix(),
        'eval_dataset_versioned': get_dataset_with_version(dataset_dir, PROJECT),
        'eval_checkpoint':        checkpoint,
        'eval_mode':              mode,
    }

    wb_logger = WandbLogger(project=project, name=run_id, id=run_id, tags=['eval'], config=eval_config, entity='armlab')
    trainer = pl.Trainer(gpus=1, enable_model_summary=False, logger=wb_logger)

    data_module = MDEDataModule(dataset_dir, batch_size=batch_size, take=take, skip=skip, test_mode=mode)

    metrics = trainer.test(model, data_module, verbose=False)
    wandb.finish()

    print(f'run_id: {run_id}')
    for metrics_i in metrics:
        for k, v in metrics_i.items():
            print(f"{k:20s}: {v:0.5f}")

    return metrics


def viz_main(dataset_dir: pathlib.Path,
             checkpoint,
             mode: str,
             user: str,
             skip: Optional[int] = None,
             project=PROJECT,
             **kwargs):
    model = load_model_artifact(checkpoint, MDE, project, version='best', user=user)
    model.training = False

    dataset = TorchMDEDataset(fetch_mde_dataset(dataset_dir), mode=mode)

    dataset = dataset_skip(dataset, skip)

    s = model.scenario

    dataset_anim = RvizAnimationController(n_time_steps=len(dataset), ns='trajs')
    time_anim = RvizAnimationController(n_time_steps=2)

    n_examples_visualized = 0
    while not dataset_anim.done:
        inputs = dataset[dataset_anim.t()]

        inputs_batch = torchify(add_batch(inputs))
        predicted_error = model(inputs_batch)

        # for only showing missclassifications:
        threshold = 0.08
        pred_close = predicted_error[0].detach().numpy() < threshold
        true_close = inputs['error'][1] < threshold
        if pred_close == true_close:
            predicted_error = remove_batch(predicted_error)
            time_anim.reset()
            while not time_anim.done:
                t = time_anim.t()
                init_viz_env(s, inputs, t)
                dataset.transition_viz_t()(s, inputs, t)
                s.plot_pred_error_rviz(predicted_error)
                time_anim.step()
                n_examples_visualized += 1

        dataset_anim.step()

    print(f"{n_examples_visualized:=}")
