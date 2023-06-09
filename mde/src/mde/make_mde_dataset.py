import pathlib
from multiprocessing import get_context

import numpy as np
from tqdm import tqdm

from link_bot_data.dataset_utils import add_predicted
from link_bot_data.split_dataset import write_mode
from link_bot_data.tf_dataset_utils import write_example, index_to_filename
from link_bot_data.wandb_datasets import wandb_save_dataset
from link_bot_pycommon.serialization import my_hdump
from moonshine.filepath_tools import load_params
from moonshine.numpify import numpify
from moonshine.torch_and_tf_utils import remove_batch, add_batch
from moonshine.torchify import torchify
from state_space_dynamics.torch_dynamics_dataset import TorchDynamicsDataset
from state_space_dynamics.train_test_dynamics import load_udnn_model_wrapper


def n_seq(max_t: int):
    return int(max_t * (max_t + 1) / 2)


def make_mde_dataset(dataset_dir: pathlib.Path,
                     checkpoint: pathlib.Path,
                     outdir: pathlib.Path,
                     step):
    mde_dataset_hparams = load_params(dataset_dir)
    with_joint_positions = False if mde_dataset_hparams['scenario'] == "watering" else True
    model = load_udnn_model_wrapper(checkpoint, with_joint_positions=with_joint_positions)


    def _set_keys_hparam(mde_dataset_hparams, k1, k2):
        mde_dataset_hparams[f'{k1}_keys'] = list(
            mde_dataset_hparams['data_collection_params'][f'{k2}_description'].keys())

    mde_dataset_hparams['dataset_dir'] = dataset_dir.as_posix()
    mde_dataset_hparams['fwd_model_hparams'] = model.hparams
    mde_dataset_hparams['predicted_state_keys'] = model.state_keys
    mde_dataset_hparams['checkpoint'] = checkpoint
    mde_dataset_hparams['make_dataset_step'] = step
    _set_keys_hparam(mde_dataset_hparams, 'true_state', 'state')
    _set_keys_hparam(mde_dataset_hparams, 'state_metadata', 'state_metadata')
    _set_keys_hparam(mde_dataset_hparams, 'env', 'env')
    _set_keys_hparam(mde_dataset_hparams, 'action', 'action')

    new_hparams_filename = outdir / 'hparams.hjson'
    my_hdump(mde_dataset_hparams, new_hparams_filename.open("w"), indent=2)

    with get_context("spawn").Pool() as pool:
        results = []
        total_example_idx = 0
        scenario_cached = None
        for mode in ['test', 'val', 'train']:
            dataset = TorchDynamicsDataset(dataset_dir=dataset_dir, mode=mode)
            if scenario_cached is not None:
                model.scenario  = scenario_cached
            else:
                model.scenario = dataset.get_scenario()
                scenario_cached = None #model.scenario
            files = []

            if len(dataset) > 0:
                steps_per_traj = dataset[0][dataset.state_keys[0]].shape[0]

                for out_example in tqdm(generate_mde_examples(model, dataset, steps_per_traj, step)):
                    # NOTE: what if two modes have the example input example? can we check if we've already generated
                    #  it an skip actually re-doing the conversion to MDE and just re-use the existing one?
                    #result = write_example(outdir, out_example, total_example_idx, 'pkl') 
                    result = pool.apply_async(func=write_example, args=(outdir, out_example, total_example_idx, 'pkl'))
                    results.append(result)

                    metadata_filename = index_to_filename('.pkl', total_example_idx)
                    full_metadata_filename = outdir / metadata_filename
                    files.append(full_metadata_filename)

                    total_example_idx += 1

            write_mode(outdir, files, mode)

        # the pool won't close unless we do this
        print("Waiting while results finish writing...")
        for result in tqdm(results):
            result.get()

    print("Saving wandb dataset", outdir)
    wandb_save_dataset(outdir, project='mde')

    return outdir


def generate_mde_examples(model, dataset, steps_per_traj, step):
    horizon = 2
    scenario = dataset.get_scenario()

    state_keys = dataset.state_keys + dataset.state_metadata_keys
  
    for example in dataset:
        from link_bot_data.tf_dataset_utils import deserialize_scene_msg
        deserialize_scene_msg(example)
        if 'time_mask' in example:
            time_mask = example['time_mask']
        else:
            time_mask = np.ones_like(example['time_idx'])
        for start_t in range(0, steps_per_traj - horizon + 1, step):
            start_state = {k: example[k][start_t:start_t + 1] for k in state_keys}  # :+1 to keep time dimension
            actions_from_start_t = {k: example[k][start_t:] for k in dataset.action_keys}

            inputs_from_start_t = {}
            inputs_from_start_t.update(start_state)
            inputs_from_start_t.update(actions_from_start_t)
            if 'scene_msg' in example:
                inputs_from_start_t['scene_msg'] = example['scene_msg']
            if 'joint_positions' in example:
                inputs_from_start_t['joint_positions'] = example['joint_positions'][start_t:start_t + 1]
                inputs_from_start_t['joint_names'] = example['joint_names'][start_t:start_t + 1]
            _inputs_from_start_t = torchify(add_batch(inputs_from_start_t))
            predictions_from_start_t = model(_inputs_from_start_t)
            predictions_from_start_t = numpify(remove_batch(predictions_from_start_t))

            actual_states_from_start_t = {k: example[k][start_t:] for k in state_keys}
            environment = {k: example[k] for k in dataset.env_keys}

            for dt in range(0, steps_per_traj - start_t - 1):
                out_example = {
                    'traj_idx': example['traj_idx'],
                    'start_t':  start_t,
                    'end_t':    start_t + dt,
                }
                out_example.update(environment)

                # add predictions to out example
                predictions_dt = {k: v[dt:dt + horizon] for k, v in predictions_from_start_t.items()}
                actual_dt = {k: v[dt:dt + horizon] for k, v in actual_states_from_start_t.items()}
                action_dt = {k: v[dt:dt + horizon - 1] for k, v in actions_from_start_t.items()}

                out_example.update(actual_dt)
                out_example.update(action_dt)
                out_example.update({add_predicted(k): v for k, v in predictions_dt.items()})

                # add error
                error = scenario.classifier_distance(actual_dt, predictions_dt)

                # store it in the metadata for faster lookup later
                out_example['metadata'] = {
                    'error': error,
                }

                if time_mask[start_t + dt] == 0 or time_mask[start_t + dt + 1] == 0:  # some examples are invalid/padding, so ignore those
                    continue

                yield out_example
