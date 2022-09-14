#!/usr/bin/env python
import pathlib

from link_bot_planning.eval_online_utils import evaluate_online_iter_outdir, get_dynamics_and_mde
from link_bot_planning.planning_evaluation import load_planner_params, evaluate_planning
from link_bot_planning.test_scenes import get_all_scene_indices
from moonshine.filepath_tools import load_hjson


def evaluate_online_iter(planner_params_filename, online_dir, iter_idx: int,  trials, seed, on_exception,
                         verbose, yes, record: bool, additional_constraints=None):
    if additional_constraints is None:
        additional_constraints = []
    online_learning_log = load_hjson(online_dir / 'logfile.hjson')

    outdir = evaluate_online_iter_outdir(planner_params_filename, online_dir, iter_idx)

    if outdir.exists() and not yes:
        k = input(f"{outdir.as_posix()} exists, do you want to resume? [Y/n]")
        if k in ['n', 'N']:
            print("Aborting")
            return

    dynamics, mde = get_dynamics_and_mde(online_learning_log, iter_idx)

    planner_params = load_planner_params(planner_params_filename)
    planner_params['online_iter'] = iter_idx
    planner_params['method_name'] = outdir.name
    planner_params['fwd_model_dir'] = dynamics
    if mde is None:
        planner_params["classifier_model_dir"] = [pathlib.Path(
            "cl_trials/new_feasibility_baseline/none")] + additional_constraints
    else:
        planner_params["classifier_model_dir"] = [mde, pathlib.Path(
            "cl_trials/new_feasibility_baseline/none")] + additional_constraints

  
    evaluate_planning(planner_params=planner_params,
                      outdir=outdir,
                      trials=trials,
                      record=record,
                      how_to_handle=on_exception,
                      verbose=verbose,
                      seed=seed)
