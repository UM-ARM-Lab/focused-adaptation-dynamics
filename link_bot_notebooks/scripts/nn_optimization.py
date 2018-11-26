#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy as np

from link_bot_notebooks import notebook_finder
from link_bot_notebooks import toy_problem_optimization_common as tpo
from link_bot_notebooks.nn_model import NNModel
from link_bot_notebooks.linear_tf_model import LinearTFModel

DIMENSIONS = {
    # Reduction
    'F1_n': 28,
    'F2_n': 16,
    # Dynamics
    'T1_n': 16,
    'T2_n': 16
}


def train(args):
    model = LinearTFModel(vars(args), N=6, M=2, L=2)

    # goal = np.array([[0], [0], [0], [1], [0], [2]])
    goals = []
    for r in np.random.randn(500, 4):
        x = r[0] * 5
        y = r[1] * 5
        theta1 = r[2] * np.pi / 2
        theta2 = r[3] * np.pi / 2
        x1 = x + np.cos(theta1)
        y1 = y + np.sin(theta1)
        x2 = x1 + np.cos(theta2)
        y2 = y1 + np.sin(theta2)
        g = np.array([[x], [y], [x1], [y1], [x2], [y2]])
        goals.append(g)

    model.setup()
    for goal in goals:
        n, x, y = tpo.load_train_test(args.dataset, N=6, M=2, L=2, g=goal, extract_func=tpo.two_link_pos_vel_extractor)
        # model = NNModel(args, N=6, M=2, L=2, dims=DIMENSIONS)
        interrupted = model.train(x, y, args.epochs)
        if interrupted:
            break


def model_only(args):
    # model = NNModel(args, N=6, M=2, L=2, dims=DIMENSIONS)
    model = LinearTFModel(vars(args), N=6, M=2, L=2)
    if args.log:
        model.init()
        model.save()


def evaluate(args):
    goal = np.array([[0], [0], [0], [1], [0], [2]])
    n, x, y = tpo.load_train_test(args.dataset, N=6, M=2, L=2, g=goal, extract_func=tpo.two_link_pos_vel_extractor)
    # model = NNModel(args, N=6, M=2, L=2, dims=DIMENSIONS)
    model = LinearTFModel(vars(args), N=6, M=2, L=2)
    model.load()
    model.evaluate(x, y)
    s = np.random.randint(-5, 5, size=(6, 1))
    print("Goal:", goal)
    print("Goal o:", model.reduce(goal))
    print("some random s:", s)
    print("o:", model.reduce(s))


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("dataset", help="dataset (txt file)")
    train_subparser.add_argument("--log", "-l", action="store_true", help="save/log the graph and summaries")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=100)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=-1)
    train_subparser.add_argument("--print-period", "-p", type=int, default=100)
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (txt file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.set_defaults(func=evaluate)

    model_only_subparser = subparsers.add_parser("model_only")
    model_only_subparser.add_argument("--log", "-l", action="store_true", help="save/log the graph and summaries")
    model_only_subparser.set_defaults(func=model_only)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)
