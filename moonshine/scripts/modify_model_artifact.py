import pathlib

import wandb
from pytorch_lightning import Trainer

from arc_utilities.algorithms import nested_dict_update
from link_bot_pycommon.load_wandb_model import get_model_artifact
from mde.torch_mde import MDE


def main():
    project = 'mde'
    checkpoint = "mde_nog_plus_planning_sdf-bfqih"
    old_best_artifact = get_model_artifact(checkpoint, project, 'armlab', 'best')
    old_latest_artifact = get_model_artifact(checkpoint, project, 'armlab', 'latest')
    artifact_dir = old_best_artifact.download()
    local_ckpt_path = pathlib.Path(artifact_dir) / "model.ckpt"
    print(f"Found {local_ckpt_path.as_posix()}")

    # model = UDNN.load_from_checkpoint(local_ckpt_path.as_posix(), train_dataset=None)
    model = MDE.load_from_checkpoint(local_ckpt_path.as_posix())
    hparams_update = {
        'env_only_sdf': True
    }
    nested_dict_update(model.hparams, hparams_update)

    trainer = Trainer()
    trainer.model = model
    trainer.save_checkpoint(local_ckpt_path)  # overwrite the old local ckpt file

    new_artifact = wandb.Artifact(name=f'model-{checkpoint}', type='model')
    new_artifact.metadata = old_best_artifact.metadata
    new_artifact.add_file(local_ckpt_path.as_posix())
    new_artifact.save(project, settings={'entity': 'armlab'})
    new_artifact.wait()
    new_artifact.aliases.append('best')
    print(f"Version: {new_artifact.version}")

    # The previous save op will automatically make it the latest version, but we don't want that. So now we restore
    # the 'latest' alias
    old_latest_artifact.aliases.append("latest")
    old_latest_artifact.save()
    old_latest_artifact.wait()

    # delete it after saving
    local_ckpt_path.unlink()


if __name__ == '__main__':
    main()
