#!/usr/bin/env python
import argparse
import pathlib

from analysis.combine_videos import combine_videos_for_iter, NoVideoError
from link_bot_pycommon.args import int_set_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', type=pathlib.Path, nargs='+')
    parser.add_argument('--episodes', type=int_set_arg, default="0-19")

    args = parser.parse_args()

    for root in args.roots:
        for episode in args.episodes:
            try:
                combined_video = combine_videos_for_iter(root, episode)
                outfilename = root / f'episode{episode:04d}_combined.mp4'
                combined_video.write_videofile(outfilename.as_posix(), fps=6)
            except NoVideoError:
                pass


if __name__ == '__main__':
    main()
