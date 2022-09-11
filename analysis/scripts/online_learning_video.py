#!/usr/bin/env python
import argparse
import pathlib
import re

import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

from analysis.combine_videos import remove_boring_frames
from moonshine.filepath_tools import load_hjson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=pathlib.Path)

    np.set_printoptions(suppress=True, precision=2)

    args = parser.parse_args()

    all_videos = []

    iter_dirs = list((args.root / 'planning_results').iterdir())
    for iter_dir in iter_dirs:
        videos = video_for_iter(iter_dir)
        all_videos.extend(videos)

    outfilename = args.root / f'online_learning.mp4'
    videos = concatenate_videoclips(all_videos)
    videos.write_videofile(outfilename.as_posix(), fps=6)


def video_for_iter(iter_dir: pathlib.Path):
    method_name_map = {
        'all_data_no_mde': 'AllDataNoMDE',
        'adaptation':      'Adaptation (ours)',
    }
    speed = 10
    w = 1080

    m = re.search(r"iteration_(\d+)", iter_dir.name)
    iter_idx = int(m.group(1))

    metadata = load_hjson(iter_dir / 'metadata.hjson')
    method_name = metadata['planner_params']['method_name']
    stylized_method_name = method_name_map[method_name]

    videos = []
    for episode in range(15):
        print(f'{iter_idx=} {episode=}')
        episode_video = edited_episode_video(episode, iter_dir, speed)
        episode_video_w_text = add_text(episode, episode_video, iter_idx, stylized_method_name, w)
        videos.append(episode_video_w_text)

    return videos


def edited_episode_video(episode, iter_dir, speed):
    latest_video_filenames = {}
    video_filenames = sorted(iter_dir.glob(f"{episode:04d}*.mp4"))
    for v in video_filenames:
        m = re.search(f'{episode:04d}-(\d\d\d\d)-(\d+)', v.as_posix())
        if m:
            attempt_idx = int(m.group(1))
            latest_video_filenames[attempt_idx] = v
    latest_video_filenames = dict(sorted(latest_video_filenames.items()))
    attempt_clips = []
    for iteration_video_filename in latest_video_filenames.values():
        attempt_clip = VideoFileClip(iteration_video_filename.as_posix(), audio=False)
        attempt_clips.append(attempt_clip)
    episode_video = concatenate_videoclips(attempt_clips)
    episode_video = remove_boring_frames(episode_video)
    episode_video = episode_video.speedx(speed)
    return episode_video


def add_text(episode, episode_video, iter_idx, stylized_method_name, w):
    text = f'{stylized_method_name} {iter_idx=} {episode=}'
    text_clip = TextClip(text,
                         font='Ubuntu-Bold',
                         fontsize=64,
                         color='white')
    h = episode_video.h
    text_clip = text_clip.set_pos((w / 2 - text_clip.w / 2, h - 10))
    size = (episode_video.w, episode_video.h + text_clip.h)
    episode_video_w_text = CompositeVideoClip([episode_video, text_clip], size=size)
    episode_video_w_text = episode_video_w_text.set_duration(episode_video.duration)
    return episode_video_w_text


if __name__ == '__main__':
    main()
