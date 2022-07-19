#!/usr/bin/env python
import argparse
import pathlib

from colorama import Fore

from arc_utilities import ros_init
from link_bot_data.new_dataset_utils import fetch_udnn_dataset
from link_bot_data.wandb_datasets import wandb_save_dataset
from link_bot_planning.planning_evaluation import evaluate_planning
from link_bot_planning.results_to_dynamics_dataset import ResultsToDynamicsDataset
from link_bot_pycommon.job_chunking import JobChunker
from mde import train_test_mde
from mde.make_mde_dataset import make_mde_dataset
from moonshine.filepath_tools import load_hjson
from state_space_dynamics import train_test_dynamics


@ros_init.with_ros("save_as_test_scene")
def main():
    """
    ./scripts/run_online_rope_sim.py results/sim_rope_adapted
    // prompts you for:
    //  - which method/baseline to run
    //  - unadapted run-id
    //  - number of iterations
    // has default args:
    //  - planner_configs/val_car/mde.hjson
    // - test_scenes/car4_alt
    // internally this will run these steps in a loop:
    //  - planning_evaluation
    //  - train_test_dynamics
    //  - train_test_mde
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("nickname")
    parser.add_argument("--on-exception", default='raise')

    args = parser.parse_args()

    root = pathlib.Path("results")
    outdir = root / args.nickname
    outdir.mkdir(exist_ok=True, parents=True)
    print(Fore.YELLOW + "Output directory: {}".format(outdir) + Fore.RESET)

    logfile_name = root / args.nickname / 'logfile.hjson'
    job_chunker = JobChunker(logfile_name)

    method_name = job_chunker.load_prompt('method_name')
    unadapted_run_id = job_chunker.load_prompt('unadapted_run_id')
    seed = int(job_chunker.load_prompt('seed'))
    planner_params_filename = job_chunker.load_prompt_filename('planner_params_filename',
                                                               'planner_configs/val_car/mde.hjson')
    planner_params = load_hjson(planner_params_filename)
    test_scenes_dir = job_chunker.load_prompt_filename('test_scenes_dir', 'test_scenes/car4_alt')
    iterations = int(job_chunker.load_prompt('iterations', 100))

    if method_name == 'adaptation':
        dynamics_params_filename = pathlib.Path("hparams/iterative_lowest_error_soft.hjson")
    elif method_name == 'all_data':
        dynamics_params_filename = pathlib.Path("hparams/all_data.hjson")
    elif method_name == 'no_adaptation':
        dynamics_params_filename = None
    else:
        raise NotImplementedError(f'Unknown method name {method_name}')

    mde_params_filename = pathlib.Path("hparams/rope.hjson")

    # initialize with unadapted model
    dynamics_dataset_dirs = []
    mde_dataset_dirs = []
    for i in range(iterations):
        sub_chunker_i = job_chunker.sub_chunker(f'iter{i}')
        planning_job_chunker = sub_chunker_i.sub_chunker("planning")

        classifiers = [pathlib.Path("cl_trials/new_feasibility_baseline/none")]
        if i != 0:
            prev_sub_chunker = job_chunker.sub_chunker(f'iter{i - 1}')
            prev_mde_run_id = prev_sub_chunker.get("mde_run_id")
            prev_dynamics_run_id = prev_sub_chunker.get("dynamics_run_id")
            classifiers.append(f'p:{prev_mde_run_id}')
        else:
            prev_dynamics_run_id = unadapted_run_id

        planning_outdir = planning_job_chunker.get('planning_outdir')
        if planning_outdir is None:
            planning_outdir = outdir / 'planning_results' / f'iteration_{i}'
            planning_outdir.mkdir(exist_ok=True, parents=True)
            planner_params["classifier_model_dir"] = classifiers
            planner_params['fwd_model_dir'] = f'p:{prev_dynamics_run_id}'
            evaluate_planning(planner_params=planner_params,
                              job_chunker=planning_job_chunker,
                              outdir=planning_outdir,
                              test_scenes_dir=test_scenes_dir,
                              seed=seed,
                              how_to_handle=args.on_exception)
            planning_job_chunker.store_result('planning_outdir', planning_outdir)

        # convert the planning results to a dynamics dataset
        dynamics_dataset_name = sub_chunker_i.get("dynamics_dataset_name")
        if dynamics_dataset_name is None:
            r = ResultsToDynamicsDataset(results_dir=planning_outdir,
                                         outname=f'{args.nickname}_dynamics_dataset_{i}',
                                         traj_length=args.traj_length,
                                         visualize=args.visualize)
            dynamics_dataset_dir_i = r.run()
            wandb_save_dataset(dynamics_dataset_dir_i, project='udnn')
            dynamics_dataset_name = dynamics_dataset_dir_i.name
            sub_chunker_i.store_result('dynamics_dataset_name', dynamics_dataset_name)

        dynamics_dataset_dirs.append(dynamics_dataset_name)

        dynamics_run_id = sub_chunker_i.get(f"dynamics_run_id")
        if dynamics_run_id is None:
            if dynamics_params_filename is not None:
                dynamics_run_id = train_test_dynamics.fine_tune_main(dataset_dir=dynamics_dataset_dirs,
                                                                     checkpoint=prev_dynamics_run_id,
                                                                     params_filename=dynamics_params_filename,
                                                                     batch_size=32,
                                                                     epochs=100,
                                                                     seed=seed,
                                                                     nickname=f'{args.nickname}_udnn_{i}',
                                                                     user='armlab')
                sub_chunker_i.store_result(f"dynamics_run_id", dynamics_run_id)

        mde_dataset_outdir = sub_chunker_i.get('mde_dataset_outdir')
        if mde_dataset_outdir is None:
            # convert the most recent dynamics dataset to and MDE dataset
            mde_dataset_outdir = root / 'mde_datasets' / f'iteration_{i}'
            mde_dataset_outdir.mkdir(parents=True, exist_ok=True)
            make_mde_dataset(dataset_dir=fetch_udnn_dataset(args.dataset_dir),
                             checkpoint=dynamics_run_id,
                             outdir=mde_dataset_outdir,
                             step=999)
            sub_chunker_i.store_result('mde_dataset_outdir', mde_dataset_outdir)

        mde_run_id = sub_chunker_i.get('mde_run_id')
        if mde_run_id is None:
            mde_run_id = train_test_mde.train_main(dataset_dir=mde_dataset_dirs,
                                                   params_filename=mde_params_filename,
                                                   batch_size=4,
                                                   epochs=100,
                                                   seed=seed,
                                                   user='armlab',
                                                   nickname=f'{args.nickname}_mde_{i}')
            sub_chunker_i.store_result('mde_run_id', mde_run_id)


if __name__ == '__main__':
    main()
