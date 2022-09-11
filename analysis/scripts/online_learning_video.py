#!/usr/bin/env python
import argparse
import pathlib
import re

import numpy as np
from more_itertools import chunked
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=pathlib.Path)

    np.set_printoptions(suppress=True, precision=2)

    args = parser.parse_args()

    speed = 10
    w = 1080

    method_name_map = {
        'all_data_no_mde': 'AllDataNoMDE',
        'adaptation':      'Adaptation (ours)',
    }

    all_videos = []

    for iter_dir in (args.root / 'planning_results').iterdir():
        m = re.search(r"iteration_(\d+)", iter_dir.name)
        iter_idx = int(m.group(1))
        print(f'{iter_idx=}')

        metadata = load_hjson(iter_dir / 'metadata.hjson')
        method_name = metadata['planner_params']['method_name']

        for episode in range(10):
            print(f'{episode=}')
            latest_video_filenames = {}

            results = load_gzipped_pickle(iter_dir / f'{episode}_metrics.pkl.gz')
            timestamps = np.array(results.get('recording_timestamps', []))
            timestamps = timestamps - timestamps[0]

            video_filenames = sorted(iter_dir.glob(f"{episode:04d}*.mp4"))
            for v in video_filenames:
                m = re.search(f'{episode:04d}-(\d\d\d\d)-(\d+)', v.as_posix())
                if m:
                    attempt_idx = int(m.group(1))
                    pq = results['planning_queries'][attempt_idx - 1]
                    attempt_start_time = pq.trial_start_time_seconds + 1659801077.9243834
                    latest_video_filenames[attempt_idx] = (attempt_start_time, v)

            if len(latest_video_filenames) == 0:
                continue

            latest_video_filenames = dict(sorted(latest_video_filenames.items()))

            method_iteration_clips = []
            for (start_time, iteration_video_filename) in latest_video_filenames.values():
                method_iteration_clip = VideoFileClip(iteration_video_filename.as_posix(), audio=False)


                method_iteration_clips.append(method_iteration_clip)
            episode_video = concatenate_videoclips(method_iteration_clips)

            action_clips = []
            for (start_t, end_t) in chunked(timestamps, 2):
                if end_t < episode_video.duration:
                    action_clip = episode_video.subclip(start_t, end_t)
                    action_clips.append(action_clip)
                else:
                    print(f"issue with iter {iter_idx} episode {episode}")
                    continue

            episode_video = concatenate_videoclips(action_clips)
            if episode_video.duration < 0:
                print(f"issue with iter {iter_idx} episode {episode}")
                continue

            episode_video = episode_video.speedx(speed)

            stylized_method_name = method_name_map[method_name]
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

            all_videos.append(episode_video_w_text)
    outfilename = args.root / f'online_learning.mp4'
    videos = concatenate_videoclips(all_videos)
    videos.write_videofile(outfilename.as_posix(), fps=6)


if __name__ == '__main__':
    main()
