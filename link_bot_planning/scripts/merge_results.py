#!/usr/bin/env python
import argparse
import pathlib
import shutil

import colorama
import hjson

from analysis.combine_videos import get_attempt_video_filenames
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=pathlib.Path)
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    data_file_extension = ".pkl.gz"
    metadata_filename = 'metadata.hjson'

    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        path = args.indir / metadata_filename
        new_path = args.outdir / metadata_filename
        # log this operation in the params!
        hparams = load_hjson(path)
        hparams['created_by_merging'] = hparams.get('created_by_merging', []) + [args.indir.as_posix()]
        hjson.dump(hparams, new_path.open('w'), indent=2)
        print(path, '-->', new_path)

    trajs_to_add = len(list(args.indir.glob("*" + data_file_extension)))
    new_traj_idx_start = len(list(args.outdir.glob("*" + data_file_extension)))

    for old_traj_idx in range(trajs_to_add):
        # copy the data/results files
        new_traj_idx = new_traj_idx_start + old_traj_idx
        old_filename = index_to_metrics_filename(data_file_extension, old_traj_idx)
        old_path = args.indir / old_filename
        new_filename = index_to_metrics_filename(data_file_extension, new_traj_idx)
        new_path = args.outdir / new_filename
        safe_copy(args.dry_run, old_path, new_path)

        metrics_filename = args.indir / f'{old_traj_idx}_metrics.pkl.gz'
        results = load_gzipped_pickle(metrics_filename)
        attempt_video_filenames = get_attempt_video_filenames(old_traj_idx, args.indir, results)

        # also copy the videos
        for old_video_attempt_path in attempt_video_filenames:
            new_video_attempt_name = f"{new_traj_idx:04d}" + old_video_attempt_path.name[4:]
            new_video_attempt_path = args.outdir / new_video_attempt_name
            safe_copy(args.dry_run, old_video_attempt_path, new_video_attempt_path)


def safe_copy(dry_run, old_path, new_path):
    print(old_path, '-->', new_path)
    if new_path.exists():
        print(f"refusing to override existing file {new_path.as_posix()}")
    else:
        if not dry_run:
            shutil.copyfile(old_path, new_path)


def index_to_metrics_filename(file_extension, traj_idx):
    new_filename = f"{traj_idx}_metrics{file_extension}"
    return new_filename


if __name__ == '__main__':
    main()
