import argparse
import pathlib

from analysis.combine_videos import combine_videos_for_iter, NoVideoError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', type=pathlib.Path, nargs='+')

    args = parser.parse_args()

    for root in args.roots:
        for episode in [1, ]:
            try:
                combined_video = combine_videos_for_iter(root, episode)
                outfilename = root / f'episode{episode:04d}_combined.mp4'
                combined_video.write_videofile(outfilename.as_posix(), fps=6)
            except NoVideoError:
                pass


if __name__ == '__main__':
    main()
