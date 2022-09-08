import pathlib
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import TextClip, VideoClip, ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

from moonshine.filepath_tools import load_hjson


class NoVideoError(Exception):
    pass


def hold_start(clip, duration):
    clip.get_frame(0)
    start = clip.subclip(clip.duration - 0.001, clip.duration).speedx(final_duration=duration)
    return start


def hold_end(clip, duration):
    clip.get_frame(-1)
    end = clip.subclip(clip.duration - 0.001, clip.duration).speedx(final_duration=duration)
    return end


def add_holds(clip):
    return hold_end(hold_start(clip, 1), 1)


def remove_boring_frames(method_iteration_clip: VideoClip):
    # for each frame, compute the naive pixel-space distance to the previous frame
    prev_frame_filtered = None
    clips = []
    ds = []
    for curr_frame in method_iteration_clip.iter_frames():
        hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        lower = np.array([110, 50, 50])
        upper = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        curr_frame_filtered = cv2.bitwise_and(curr_frame, curr_frame, mask=mask)

        clips.append(ImageClip(curr_frame_filtered).set_duration(1 / 6))

        if prev_frame_filtered is not None:
            delta = (curr_frame_filtered - prev_frame_filtered).sum(-1)
            plt.imshow(delta)
            plt.show()
            d = np.linalg.norm(delta)
            ds.append(d)
            pass
        prev_frame_filtered = curr_frame_filtered

    concat_clip = concatenate_videoclips(clips)
    plt.plot(ds)
    plt.show()

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
        method_iteration_clip = method_iteration_clip.speedx(speed)
        method_iteration_clip = add_holds(method_iteration_clip)
        method_iteration_clips.append(method_iteration_clip)
    method_iteration_video = concatenate_videoclips(method_iteration_clips)

    method_name_txt = TextClip(method_name,
                               font='Ubuntu-Bold',
                               fontsize=64,
                               color='white')

    h = method_iteration_video.h
    method_name_txt = method_name_txt.set_pos((w / 2 - method_name_txt.w / 2, h - 10))
    size = (method_iteration_video.w, method_iteration_video.h + method_name_txt.h)
    method_iteration_video_w_txt = method_iteration_video
    # method_iteration_video_w_txt = CompositeVideoClip([method_iteration_video, method_name_txt], size=size)
    # method_iteration_video_w_txt = method_iteration_video_w_txt.set_duration(method_iteration_video.duration)

    return method_iteration_video_w_txt
