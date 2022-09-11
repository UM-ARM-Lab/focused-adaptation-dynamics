import pathlib
import re

import cv2
import moviepy
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import TextClip, VideoClip, ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

from moonshine.filepath_tools import load_hjson


class NoVideoError(Exception):
    pass


def add_holds(clip):
    clip = moviepy.video.fx.all.freeze(clip, t='end', freeze_duration=1)
    clip = moviepy.video.fx.all.freeze(clip, t=0, freeze_duration=1)
    return clip


def remove_boring_frames(method_iteration_clip: VideoClip):
    # for each frame, compute the naive pixel-space distance to the previous frame
    prev_frame = None
    clips = []
    for curr_frame in method_iteration_clip.iter_frames():
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


def combine_videos_for_iter(root: pathlib.Path, episode: int, w=1080, speed=10):
    metadata = load_hjson(root / 'metadata.hjson')
    method_name = metadata['planner_params']['method_name']

    video_filenames = sorted(root.glob("*.mp4"))

    latest_video_filenames = {}
    # iterate over possible video files and keep the latest one for each attempt idx
    for v in video_filenames:
        m = re.search(f'{episode:04d}-(\d\d\d\d)-\d+', v.as_posix())
        if m:
            attempt_idx = int(m.group(1))
            latest_video_filenames[attempt_idx] = v

    if len(latest_video_filenames) == 0:
        raise NoVideoError(f"No videos for {root} episode {episode}")
    latest_video_filenames = dict(sorted(latest_video_filenames.items()))

    method_iteration_clips = []
    for iteration_video_filename in latest_video_filenames.values():
        method_iteration_clip = VideoFileClip(iteration_video_filename.as_posix(), audio=False)
        method_iteration_clips.append(method_iteration_clip)
    method_iteration_video = concatenate_videoclips(method_iteration_clips)
    method_iteration_video = method_iteration_video.speedx(speed)
    method_iteration_video = add_holds(method_iteration_video)

    method_name_txt = TextClip(method_name,
                               font='Ubuntu-Bold',
                               fontsize=64,
                               color='white')

    h = method_iteration_video.h
    method_name_txt = method_name_txt.set_pos((w / 2 - method_name_txt.w / 2, h - 10))
    size = (method_iteration_video.w, method_iteration_video.h + method_name_txt.h)
    method_iteration_video_w_txt = CompositeVideoClip([method_iteration_video, method_name_txt], size=size)
    method_iteration_video_w_txt = method_iteration_video_w_txt.set_duration(method_iteration_video.duration)

    return method_iteration_video_w_txt
