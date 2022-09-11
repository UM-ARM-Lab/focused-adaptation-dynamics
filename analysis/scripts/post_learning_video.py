#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
from moviepy.editor import *  # you actually have to do this or seemingly random functions won't exist

from analysis.combine_videos import video_for_post_learning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=pathlib.Path)

    np.set_printoptions(suppress=True, precision=2)

    args = parser.parse_args()

    videos = video_for_post_learning(args.root)

    outfilename = args.root / f'post_learning.mp4'
    videos = concatenate_videoclips(videos)
    videos.write_videofile(outfilename.as_posix(), fps=6)


if __name__ == '__main__':
    main()
