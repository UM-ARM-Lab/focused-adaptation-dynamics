#!/usr/bin/env python
import argparse
import logging
import pathlib

import colorama
import numpy as np
import tensorflow as tf

from arc_utilities import ros_init
from link_bot_data.dataset_utils import make_unique_outdir
from link_bot_planning.planning_evaluation import evaluate_multiple_planning, load_planner_params
from link_bot_planning.test_scenes import get_all_scene_indices
from link_bot_pycommon.args import int_set_arg
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(None)


@ros_init.with_ros("planning_evaluation")
def main():
    colorama.init(autoreset=True)
    np.set_printoptions(suppress=True, precision=5, linewidth=250)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('planner_params', type=pathlib.Path, help='planner params hjson file')
    parser.add_argument("test_scenes_dir", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path, help='used in making the output directory')
    parser.add_argument("--trials", type=int_set_arg)
    parser.add_argument("--timeout", type=int, help='timeout to override what is in the planner config file')
    parser.add_argument("--seed", type=int, help='an additional seed for testing randomness', default=0)
    parser.add_argument("--no-execution", action="store_true", help='no execution')
    parser.add_argument("--on-exception", choices=['raise', 'catch', 'retry'], default='retry')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--record', action='store_true', help='record')
    parser.add_argument('--fast', action='store_true', help='for real val this will remove the sleeps')
    parser.add_argument('--dynamics', type=pathlib.Path)
    parser.add_argument('--classifier', type=pathlib.Path)
    parser.add_argument('--recovery', type=pathlib.Path)
    parser.add_argument('--continue-from', type=pathlib.Path)

    args = parser.parse_args()

    if args.continue_from is not None:
        print(f"Ignoring nickname {args.outdir}")
        outdir = args.continue_from
    else:
        outdir = make_unique_outdir(args.outdir)

    planner_params = load_planner_params(args.planner_params)
    planner_params['method_name'] = args.outdir.name
    if args.dynamics:
        planner_params['fwd_model_dir'] = args.dynamics

    classifiers = [pathlib.Path("cl_trials/new_feasibility_baseline/none")]
    if args.classifier:
        classifiers.append(args.classifier)
    planner_params["classifier_model_dir"] = classifiers

    if args.recovery:
        planner_params["recovery"]["recovery_model_dir"] = args.recovery

    if not args.test_scenes_dir.exists():
        print(f"Test scenes dir {args.test_scenes_dir} does not exist")
        return

    if args.trials is None:
        args.trials = list(get_all_scene_indices(args.test_scenes_dir))
        print(args.trials)

    def _on_scenario_cb(scenario):
        if args.fast and hasattr(scenario, 'fast'):
            print("SPEEEEEEEED!")
            scenario.fast = True

    evaluate_multiple_planning(outdir=outdir,
                               planners_params=[(args.planner_params.stem, planner_params)],
                               trials=args.trials,
                               how_to_handle=args.on_exception,
                               use_gt_rope=True,
                               verbose=args.verbose,
                               timeout=args.timeout,
                               test_scenes_dir=args.test_scenes_dir,
                               no_execution=args.no_execution,
                               logfile_name=None,
                               record=args.record,
                               seed=args.seed,
                               on_scenario_cb=_on_scenario_cb,
                               )


if __name__ == '__main__':
    main()
