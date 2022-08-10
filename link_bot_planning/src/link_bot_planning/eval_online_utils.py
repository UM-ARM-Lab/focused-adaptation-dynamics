import pathlib


def evaluate_online_iter_outdir(planner_params: pathlib.Path, online_dir: pathlib.Path):
    planning_params_name = planner_params.stem
    nickname = f"{online_dir.name}_iter{iter}-{planning_params_name}"
    outdir = pathlib.Path(f"/media/shared/planning_results/{nickname}")
    return outdir


def get_dynamics_and_mde(log, i: int):
    iter_log = log[f'iter{i}']
    dynamics_run_id = iter_log['dynamics_run_id']
    mde_run_id = iter_log.get('mde_run_id', None)
    if mde_run_id is None:
        return f'p:{dynamics_run_id}', None
    else:
        return f'p:{dynamics_run_id}', f'p:{mde_run_id}'