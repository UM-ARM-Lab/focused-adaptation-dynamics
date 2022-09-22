#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
from moviepy.editor import *  # you actually have to do this or seemingly random functions won't exist

from analysis.combine_videos import video_for_post_learning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=pathlib.Path)
    parser.add_argument('--final-speedup', type=int, default=4)

    np.set_printoptions(suppress=True, precision=2)

    args = parser.parse_args()

    videos = video_for_post_learning(args.root, args.final_speedup)
    for i, video_i in enumerate(videos):
        outfilename_i = args.root / f'episode_{i}.mp4'
        video_i.write_videofile(outfilename_i.as_posix(), fps=6 * args.final_speedup)

    video = concatenate_videoclips(videos)

    outfilename = args.root / f'post_learning.mp4'
    video.write_videofile(outfilename.as_posix(), fps=6 * args.final_speedup)


if __name__ == '__main__':
    main()
