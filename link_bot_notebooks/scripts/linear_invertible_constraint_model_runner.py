#!/usr/bin/env python
from __future__ import print_function

import argparse
import re
import os
import numpy as np

from link_bot_notebooks import toy_problem_optimization_common as tpo
from link_bot_notebooks import linear_invertible_constraint_model
from link_bot_notebooks import experiments_util

DT = 0.1


def train(args):
    model = linear_invertible_constraint_model.LinearInvertibleModel(vars(args), N=args.N, M=args.M, K=args.K,
                                                                     L=args.L, dt=DT)

    goals = []
    for _ in range(args.n_goals):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        theta1 = np.random.uniform(-np.pi / 2, np.pi / 2)
        theta2 = np.random.uniform(-np.pi / 2, np.pi / 2)
        x1 = x + np.cos(theta1)
        y1 = y + np.sin(theta1)
        x2 = x1 + np.cos(theta2)
        y2 = y1 + np.sin(theta2)
        g = np.array([[x], [y], [x1], [y1], [x2], [y2]])
        goals.append(g)

    model.setup()

    log_path = experiments_util.experiment_name(args.log)
    log_data = np.loadtxt(args.dataset)
    trajectory_length_during_collection = tpo.dataset_name(args.dataset)
    x = tpo.load_train2(log_data, tpo.link_pos_vel_extractor2_indeces(), trajectory_length_during_collection, 1)

    for goal in goals:
        interrupted = model.train(x, goal, args.epochs, log_path)
        if interrupted:
            break

    # evaluate
    goal = np.array([[0], [0], [0], [1], [0], [2]])
    model.evaluate(x, goal)


def model_only(args):
    model = linear_invertible_constraint_model.LinearInvertibleModel(vars(args), N=args.N, M=args.M, K=args.K,
                                                                     L=args.L, dt=DT)
    if args.log:
        model.init()
        log_path = experiments_util.experiment_name(args.log)
        full_log_path = os.path.join(os.getcwd(), "log_data", log_path)
        model.save(full_log_path)


def evaluate(args):
    goal = np.array([[0], [0], [0], [1], [0], [2]])
    x = tpo.load_train(args.dataset, N=args.N, L=args.L, extract_func=tpo.link_pos_vel_extractor2(args.N))
    model = linear_invertible_constraint_model.LinearInvertibleModel(vars(args), N=args.N, M=args.M, K=args.K,
                                                                     L=args.L, dt=0.1)
    model.load()
    model.evaluate(x, goal)


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state", type=int, default=4)
    parser.add_argument("-K", help="dimensions in latent state", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("dataset", help="dataset (txt file)")
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=100)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=1024)
    train_subparser.add_argument("--cog-period", "-p", type=int, default=100)
    train_subparser.add_argument("--n-goals", "-n", type=int, default=500)
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (txt file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.add_argument("--n-steps", "-s", type=int, default=1)
    eval_subparser.set_defaults(func=evaluate)
    eval_subparser.add_argument("--batch-size", "-b", type=int, default=4096)

    model_only_subparser = subparsers.add_parser("model_only")
    model_only_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    model_only_subparser.add_argument("--n-steps", "-s", type=int, default=1)
    model_only_subparser.set_defaults(func=model_only)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)
