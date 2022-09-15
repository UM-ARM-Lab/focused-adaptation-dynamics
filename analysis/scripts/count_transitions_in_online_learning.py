#!/usr/bin/env python
import argparse
import pathlib

from link_bot_pycommon.serialization import load_gzipped_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('online_dir', type=pathlib.Path)

    args = parser.parse_args()

    n_total_transitions = 0
    dynamics_datasets_dir = args.online_dir / 'dynamics_datasets'
    for d in dynamics_datasets_dir.iterdir():
        for data_filename in d.glob("*.pkl.gz"):
            data = load_gzipped_pickle(data_filename)
            if 'time_mask' in data:
                n_valid_transitions = int(data['time_mask'].sum())
                n_total_transitions += n_valid_transitions

    print(f"{n_total_transitions=}")


if __name__ == '__main__':
    main()
