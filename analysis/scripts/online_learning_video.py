#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
from moviepy.video.compositing.concatenate import concatenate_videoclips

from analysis.combine_videos import quick_video_for_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=pathlib.Path)

    np.set_printoptions(suppress=True, precision=2)

    args = parser.parse_args()

    all_videos = []

    iter_dirs = list((args.root / 'planning_results').iterdir())
    for iter_dir in iter_dirs:
        videos = quick_video_for_iter(iter_dir)
        all_videos.extend(videos)

    outfilename = args.root / f'online_learning.mp4'
    videos = concatenate_videoclips(all_videos)
    videos.write_videofile(outfilename.as_posix(), fps=6)


if __name__ == '__main__':
    main()
