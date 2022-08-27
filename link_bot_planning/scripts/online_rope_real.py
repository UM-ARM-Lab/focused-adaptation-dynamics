#!/usr/bin/env python
import argparse
import itertools
import pathlib
import warnings
from time import perf_counter, sleep

import numpy as np
import rospkg
from more_itertools import chunked

from link_bot_planning.planning_evaluation import evaluate_planning, load_planner_params
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import pathify
from moonshine.gpu_config import limit_gpu_mem
from moonshine.magic import wandb_lightning_magic

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    from ompl import util as ou

from colorama import Fore

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from link_bot_data.wandb_datasets import wandb_save_dataset
from link_bot_planning.results_to_dynamics_dataset import ResultsToDynamicsDataset
from link_bot_pycommon.job_chunking import JobChunker
from mde import train_test_mde
from mde.make_mde_dataset import make_mde_dataset
from state_space_dynamics import train_test_dynamics

limit_gpu_mem(None)  # just in case TF is used somewhere


@ros_init.with_ros("online_rope_real")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nickname")
    parser.add_argument("--on-exception", default='raise')

    args = parser.parse_args()

    ou.setLogLevel(ou.LOG_ERROR)
    wandb_lightning_magic()

    if '-' not in args.nickname:
        print("You forgot to put the seed in the nickname!")
        return

    root = pathlib.Path("/media/shared/real_online_adaptation")
    outdir = root / args.nickname
    outdir.mkdir(exist_ok=True, parents=True)
    print(Fore.YELLOW + "Output directory: {}".format(outdir) + Fore.RESET)

    r = rospkg.RosPack()
    dynamics_pkg_dir = pathlib.Path(r.get_path('state_space_dynamics'))
    mde_pkg_dir = pathlib.Path(r.get_path('mde'))

    logfile_name = root / args.nickname / 'logfile.hjson'
    job_chunker = JobChunker(logfile_name)

    name_no_seed, seed = args.nickname.split("-")
    name_parts = name_no_seed.split("_")
    group_name = name_parts[0]
    method_name = "_".join(name_parts[1:])
    if method_name not in args.nickname:
        print(f"{args.nickname=} doesn't make sense with {method_name=}, aborting!")
        return
    if method_name == 'adaptation':
        dynamics_params_filename = dynamics_pkg_dir / "hparams" / "iterative_lowest_error_soft_online.hjson"
        unadapted_run_id = 'sim_rope_unadapted-dme7l'
    elif method_name in ['all_data', 'all_data_no_mde']:
        dynamics_params_filename = dynamics_pkg_dir / "hparams" / "all_data_online.hjson"
        unadapted_run_id = 'sim_rope_unadapted_all_data-1lpq9'
    else:
        raise NotImplementedError(f'Unknown method name {method_name}')

    seed = int(seed)
    job_chunker.store_result('method_name', method_name)
    job_chunker.store_result('group_name', group_name)
    job_chunker.store_result('seed', seed)
    job_chunker.store_result('unadapted_run_id', unadapted_run_id)

    planner_params_filename = job_chunker.load_prompt_filename('planner_params_filename',
                                                               'planner_configs/val_car/real_online_learning.hjson')
    test_scenes_dir = job_chunker.load_prompt_filename('test_scenes_dir', 'test_scenes/car5_real')
    iterations = int(job_chunker.load_prompt('iterations', 10))
    n_trials_per_iteration = int(job_chunker.load_prompt('n_trials_per_iteration', 10))
    udnn_init_epochs = int(job_chunker.load_prompt('udnn_init_epochs', 2))
    udnn_scale_epochs = int(job_chunker.load_prompt('udnn_scale_epochs', 0.25))
    mde_init_epochs = int(job_chunker.load_prompt('mde_init_epochs', 10))
    mde_scale_epochs = int(job_chunker.load_prompt('mde_scale_epochs', 0.25))

    mde_params_filename = mde_pkg_dir / "hparams" / "rope.hjson"

    all_trial_indices = list(get_all_scene_indices(test_scenes_dir))
    trial_indices_generator = chunked(itertools.cycle(all_trial_indices), n_trials_per_iteration)
    scenario = get_scenario("real_val_with_robot_feasibility_checking", {'rope_name': 'none'})

    # initialize with unadapted model
    dynamics_dataset_dirs = []
    mde_dataset_dirs = []
    prev_mde_run_id = None
    for i in range(iterations):
        print(Fore.CYAN + f"Iteration {i}" + Fore.RESET)

        sub_chunker_i = job_chunker.sub_chunker(f'iter{i}')

        prev_mde = None
        if i != 0:
            prev_sub_chunker = job_chunker.sub_chunker(f'iter{i - 1}')
            prev_dynamics_run_id = prev_sub_chunker.get("dynamics_run_id")
            if method_name == 'all_data_no_mde':
                print("Not using an MDE!")
            else:
                prev_mde_run_id = prev_sub_chunker.get("mde_run_id")
                prev_mde = f'p:model-{prev_mde_run_id}:latest'
        else:
            prev_dynamics_run_id = unadapted_run_id

        planning_outdir = pathify(sub_chunker_i.get('planning_outdir'))
        planning_trials = next(trial_indices_generator)  # must call every time or it won't be reproducible
        if planning_outdir is None:
            t0 = perf_counter()
            planning_outdir = outdir / 'planning_results' / f'iteration_{i}'
            planning_outdir.mkdir(exist_ok=True, parents=True)

            planner_params = load_planner_params(planner_params_filename)
            planner_params['method_name'] = method_name
            planner_params['fwd_model_dir'] = f'p:model-{prev_dynamics_run_id}:latest'
            planner_params["classifier_model_dir"] = [pathlib.Path("cl_trials/new_feasibility_baseline/none")]
            if prev_mde is not None:
                planner_params["classifier_model_dir"].append(prev_mde)
            evaluate_planning(planner_params=planner_params,
                              outdir=outdir,
                              scenario=scenario,
                              verbose=1,
                              trials=planning_trials,
                              how_to_handle=args.on_exception,
                              test_scenes_dir=test_scenes_dir,
                              seed=seed)
            sub_chunker_i.store_result('planning_outdir', planning_outdir.as_posix())
            dt = perf_counter() - t0
            sub_chunker_i.store_result('planning_outdir_dt', dt)

        # convert the planning results to a dynamics dataset
        dynamics_dataset_name = sub_chunker_i.get("dynamics_dataset_name")
        if dynamics_dataset_name is None:
            t0 = perf_counter()
            r = ResultsToDynamicsDataset(results_dir=planning_outdir,
                                         outname=f'{args.nickname}_dynamics_dataset_{i}',
                                         root=outdir / 'dynamics_datasets',
                                         traj_length=100,
                                         val_split=0,
                                         test_split=0,
                                         visualize=False)
            dynamics_dataset_dir_i = r.run()
            wandb_save_dataset(dynamics_dataset_dir_i, project='udnn')
            dynamics_dataset_name = dynamics_dataset_dir_i.name
            sub_chunker_i.store_result('dynamics_dataset_name', dynamics_dataset_name)
            dt = perf_counter() - t0
            sub_chunker_i.store_result('dynamics_dataset_name_dt', dt)
            sleep(10)  # in case wandb hasn't synced yet...

        dynamics_dataset_dirs.append(dynamics_dataset_name)

        dynamics_run_id = sub_chunker_i.get(f"dynamics_run_id")
        if dynamics_run_id is None:
            if dynamics_params_filename is not None:
                t0 = perf_counter()
                dynamics_run_id = train_test_dynamics.fine_tune_main(dataset_dir=dynamics_dataset_dirs,
                                                                     checkpoint=prev_dynamics_run_id,
                                                                     params_filename=dynamics_params_filename,
                                                                     batch_size=32,
                                                                     steps=-1,
                                                                     epochs=int(
                                                                         udnn_init_epochs + i * udnn_scale_epochs),
                                                                     repeat=100,
                                                                     no_val=True,
                                                                     seed=seed,
                                                                     nickname=f'{args.nickname}_udnn_{i}',
                                                                     user='armlab',
                                                                     # extra args get stored in run config
                                                                     online=True,
                                                                     online_iter=i,
                                                                     )
                print(f'{dynamics_run_id=}')
                sub_chunker_i.store_result(f"dynamics_run_id", dynamics_run_id)
                dt = perf_counter() - t0
                sub_chunker_i.store_result('fine_tune_dynamics_dt', dt)

        mde_dataset_name = sub_chunker_i.get('mde_dataset_name')
        if mde_dataset_name is None and method_name != 'all_data_no_mde':
            t0 = perf_counter()
            mde_dataset_name = f'{args.nickname}_mde_dataset_{i}'
            mde_dataset_outdir = outdir / 'mde_datasets' / mde_dataset_name
            mde_dataset_outdir.mkdir(parents=True, exist_ok=True)
            make_mde_dataset(dataset_dir=fetch_udnn_dataset(dynamics_dataset_name),
                             checkpoint=dynamics_run_id,
                             outdir=mde_dataset_outdir,
                             step=999)
            sub_chunker_i.store_result('mde_dataset_name', mde_dataset_name)
            dt = perf_counter() - t0
            sub_chunker_i.store_result('make_mde_dataset_dt', dt)
        mde_dataset_dirs.append(mde_dataset_name)

        mde_run_id = sub_chunker_i.get('mde_run_id')
        if mde_run_id is None and method_name != 'all_data_no_mde':
            t0 = perf_counter()
            if i == 0:
                mde_run_id = train_test_mde.train_main(dataset_dir=mde_dataset_dirs,
                                                       params_filename=mde_params_filename,
                                                       batch_size=32,
                                                       steps=-1,
                                                       epochs=int(mde_init_epochs + i * mde_scale_epochs),
                                                       train_mode='all',
                                                       val_mode='all',  # yes needed env if no_val=True
                                                       no_val=True,
                                                       seed=seed,
                                                       user='armlab',
                                                       nickname=f'{args.nickname}_mde_{i}',
                                                       # extra args get stored in run config
                                                       online=True,
                                                       online_iter=i,
                                                       )
            else:
                mde_run_id = train_test_mde.fine_tune_main(dataset_dir=mde_dataset_dirs,
                                                           checkpoint=prev_mde_run_id,
                                                           params_filename=mde_params_filename,
                                                           batch_size=32,
                                                           steps=-1,
                                                           epochs=mde_init_epochs + i * mde_scale_epochs,
                                                           train_mode='all',
                                                           val_mode='all',
                                                           no_val=True,
                                                           seed=seed,
                                                           user='armlab',
                                                           nickname=f'{args.nickname}_mde_{i}',
                                                           # extra args get stored in run config
                                                           online=True,
                                                           online_iter=i,
                                                           )
            sub_chunker_i.store_result('mde_run_id', mde_run_id)
            dt = perf_counter() - t0
            sub_chunker_i.store_result('fine_tune_mde_dt', dt)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4, linewidth=200)
    main()
