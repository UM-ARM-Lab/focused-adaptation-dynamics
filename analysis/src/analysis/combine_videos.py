import pathlib
import re

import cv2
import moviepy
import numpy as np
from moviepy.editor import *  # you actually have to do this or seemingly random functions won't exist
from tqdm import trange

from analysis import results_metrics
from link_bot_pycommon.get_scenario import get_scenario_cached
from link_bot_pycommon.serialization import load_gzipped_pickle
from moonshine.filepath_tools import load_hjson


class NoVideoError(Exception):
    pass


method_name_map = {
    'all_data_no_mde': 'AllDataNoMDE',
    'adaptation':      'FOCUS (ours)',
}


def make_success_clip(clip, success: bool, duration):
    if success:
        icon_clip = ImageClip("~/Pictures/icons/check_mark.png")
    else:
        icon_clip = ImageClip("~/Pictures/icons/cross_mark.png")
    icon_clip = icon_clip.resize(width=clip.w * 0.1)
    icon_clip = icon_clip.set_pos((10, clip.h - icon_clip.h - 10))
    icon_clip.duration = duration
    clip_with_icon = CompositeVideoClip([clip, icon_clip])
    clip_with_icon.duration = duration
    return clip_with_icon


def add_holds(clip, success=None, start_freeze_duration=1.5, end_freeze_duration=0.5):
    clip = moviepy.video.fx.all.freeze(clip, t=0, freeze_duration=start_freeze_duration)
    for final_frame in clip.iter_frames():
        pass
    final_frame_clip = ImageClip(final_frame).set_duration(end_freeze_duration)
    if success is None:
        clip = concatenate_videoclips([clip, final_frame_clip])
    else:
        final_frame_clip_with_success = make_success_clip(final_frame_clip, success, duration=end_freeze_duration)
        clip = concatenate_videoclips([clip, final_frame_clip_with_success])
    return clip


def remove_similar_frames(clip: VideoClip):
    # for each frame, compute the naive pixel-space distance to the previous frame
    prev_frame = None

    clips = []
    # always include first frame
    clips.append(ImageClip(clip.get_frame(0)).set_duration(1 / 6))

    for curr_frame in clip.iter_frames():
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

        if prev_frame is not None:
            diff = curr_frame_gray.copy()
            cv2.absdiff(curr_frame_gray, prev_frame, diff)
            d = np.linalg.norm(diff)
            if d > 5000:
                clips.append(ImageClip(curr_frame).set_duration(1 / 6))
        prev_frame = curr_frame_gray

    concat_clip = concatenate_videoclips(clips)

    return concat_clip


def video_for_post_learning(iter_dir: pathlib.Path, final_speedup: int):
    internal_speedup = 10

    metadata = load_hjson(iter_dir / 'metadata.hjson')
    method_name = metadata['planner_params']['method_name']
    stylized_method_name = method_name_map.get(method_name, method_name)

    videos = []
    for episode in trange(32):
        metrics_filename = (iter_dir / f'{episode}_metrics.pkl.gz')
        if not metrics_filename.exists():
            continue
        episode_video = edited_episode_video(episode, iter_dir, internal_speedup, metrics_filename)
        text = f'Post-Learning: {stylized_method_name} {episode=} ({final_speedup * internal_speedup}x, pauses removed)'
        episode_video_w_text = add_text(episode_video, text)
        videos.append(episode_video_w_text)

    return videos


def quick_video_for_iter(iter_dir: pathlib.Path, speed: float):
    m = re.search(r"iteration_(\d+)", iter_dir.name)
    iter_idx = int(m.group(1))

    metadata = load_hjson(iter_dir / 'metadata.hjson')
    method_name = metadata['planner_params']['method_name']
    stylized_method_name = method_name_map[method_name]

    videos = []
    for metrics_filename in iter_dir.glob("*metrics.pkl.gz"):
        m = re.search('(\d+)_metrics.pkl.gz', metrics_filename.name)
        episode = int(m.group(1))
        print(f'{iter_idx=} {episode=}')
        episode_video = edited_episode_video(episode, iter_dir, speed, metrics_filename)
        text = f'{stylized_method_name} {iter_idx=} {episode=}'
        episode_video_w_text = add_text(episode_video, text)
        videos.append(episode_video_w_text)

    return videos


def edited_episode_video(episode, iter_dir, speed, metrics_filename):
    results = load_gzipped_pickle(metrics_filename)
    trial_metadata = load_hjson(metrics_filename.parent / 'metadata.hjson')
    scenario = get_scenario_cached("floating_rope", {'rope_name': ''})
    success = results_metrics.success(metrics_filename, scenario, trial_metadata, results)

    attempt_video_filenames = []
    for attempt_idx in range(len(results['steps'])):
        potential_video_filenames = sorted(iter_dir.glob(f"{episode:04d}-{attempt_idx + 1:04d}-*.mp4"))
        latest_attempt_video_filename = potential_video_filenames[0]
        for f in potential_video_filenames:
            if 'edited' not in f.name:
                latest_attempt_video_filename = f
        attempt_video_filenames.append(latest_attempt_video_filename)

    attempt_clips = []
    for iteration_video_filename in attempt_video_filenames:
        attempt_clip = load_edited_clip(iteration_video_filename)
        attempt_clips.append(attempt_clip)
    episode_video = concatenate_videoclips(attempt_clips)
    episode_video = episode_video.speedx(speed)
    episode_video = add_holds(episode_video, success)
    return episode_video


def load_edited_clip(iteration_video_filename: pathlib.Path):
    edited_clip_filename = iteration_video_filename.parent / f"{iteration_video_filename.stem}.edited.mp4"

    if not edited_clip_filename.exists():
        attempt_clip = VideoFileClip(iteration_video_filename.as_posix(), audio=False)
        edited_attempt_clip = remove_similar_frames(attempt_clip)
        edited_attempt_clip.write_videofile(edited_clip_filename.as_posix(), fps=6)

    return VideoFileClip(edited_clip_filename.as_posix())


def add_text(episode_video, text):
    # FIXME: no idea why this works, but without it, the video is black
    episode_video = episode_video.resize(width=episode_video.h)
    text_clip = TextClip(text, font='Ubuntu-Bold', fontsize=32, color='white', bg_color='black').set_position(
        ('center', 'bottom'))
    episode_video_w_text = CompositeVideoClip([episode_video, text_clip])
    episode_video_w_text.duration = episode_video.duration
    return episode_video_w_text
