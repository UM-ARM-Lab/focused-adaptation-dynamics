import argparse
import pathlib

from moviepy.editor import *
from moviepy.video.fx.freeze import freeze

from analysis.combine_videos import combine_videos_for_iter


def hold_end(clip, duration):
    clip.get_frame(-1)
    end = clip.subclip(clip.duration - 0.001, clip.duration).speedx(final_duration=duration)
    return end


def make_learning_progress_video(args):
    w = 1080
    hspacing = 1.05
    speed = 10
    iteration_videos = []
    for iteration in args.iterations:
        method_iteration_videos = make_method_iteration_videos(args.roots, iteration, speed, w)
        max_duration, method_iteration_videos_held = add_holds(method_iteration_videos)

        iter_txt = TextClip(f'Iteration {iteration + 1}',
                            font='Ubuntu-Bold',
                            fontsize=90,
                            color='white')
        iter_txt = iter_txt.set_pos((int(w - iter_txt.w / 2 + 40), 0))
        speedup_txt = TextClip(f'{speed}x',
                               font='Ubuntu-Bold',
                               fontsize=60,
                               color='white')

        first_vid = next(iter(method_iteration_videos_held.values()))
        method_iteration_videos_positioned = []
        for i, method_iteration_video_held in enumerate(method_iteration_videos_held.values()):
            method_iteration_videos_positioned.append(
                method_iteration_video_held.set_pos((int(hspacing * w * i), iter_txt.h)))

        full_size = (int(hspacing * 2 * w), int(first_vid.h + iter_txt.h))
        iteration_video = CompositeVideoClip(method_iteration_videos_positioned + [speedup_txt, iter_txt],
                                             size=full_size)
        iteration_video.duration = max_duration
        iteration_video = iteration_video.set_duration(method_iteration_videos_positioned[0].duration)
        iteration_videos.append(iteration_video)

    final = concatenate_videoclips(iteration_videos)

    outfilename = args.roots[0] / 'final.mp4'
    final.write_videofile(outfilename.as_posix(), fps=60)


def add_holds(method_iteration_videos):
    max_duration = max([v.duration for v in method_iteration_videos.values()]) + 1
    method_iteration_videos_held = {}
    for method_name, method_iteration_video in method_iteration_videos.items():
        method_iteration_video_held = freeze(method_iteration_video, t='end', total_duration=max_duration,
                                             padding_end=0.01)
        method_iteration_videos_held[method_name] = method_iteration_video_held
    return max_duration, method_iteration_videos_held


def make_method_iteration_videos(roots, iteration, speed, w):
    method_iteration_videos = {}
    for root in roots:
        with (root / 'method_name').open("r") as f:
            method_name = f.readline().strip("\n")
        method_iteration_video = make_method_iteration_video(iteration, root, method_name, speed, w)
        if method_name in method_iteration_videos:
            raise RuntimeError("duplicate method detected!")
        method_iteration_videos[method_name] = method_iteration_video
    return method_iteration_videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', type=pathlib.Path, nargs='+')

    args = parser.parse_args()

    for root in args.roots:
        for episode in [1, 2, 3, 4, 5]:
            combined_video = combine_videos_for_iter(root, episode)

            outfilename = root / f'episode{episode:04d}_combined.mp4'
            combined_video.write_videofile(outfilename.as_posix(), fps=6)


if __name__ == '__main__':
    main()
