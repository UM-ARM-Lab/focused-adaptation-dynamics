#!/usr/bin/env python
import argparse

from link_bot_gazebo import gazebo_utils
from link_bot_pycommon.args import run_subparsers


def suspend(args):
    gazebo_processes = gazebo_utils.suspend()
    print(gazebo_processes)


def resume(args):
    gazebo_processes = gazebo_utils.resume()
    print(gazebo_processes)


def statuses(args):
    for s in gazebo_utils.statuses():
        print(s)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    suspend_parser = subparsers.add_parser('suspend')
    suspend_parser.set_defaults(func=suspend)
    resume_parser = subparsers.add_parser('resume')
    resume_parser.set_defaults(func=resume)
    statuses_parser = subparsers.add_parser('statuses')
    statuses_parser.set_defaults(func=statuses)

    run_subparsers(parser)


if __name__ == '__main__':
    main()
