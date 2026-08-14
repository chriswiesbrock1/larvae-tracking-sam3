"""Cutting one video per droplet out of the raw recording.

All ROI writers are opened at once and the source video is read exactly once,
so the cost is a single decode pass regardless of the number of droplets.
Anything else that needs to look at every frame — reading the temperature
display, for example — hooks into that same pass through ``frame_callback``
rather than decoding the video a second time.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import cv2

from larvatracker.config import DEFAULT_FPS
from larvatracker.droplets import Droplet


def frame_timestamp(cap: cv2.VideoCapture, frame_idx: int, fps: float, last_time: float) -> float:
    """Best available timestamp for the frame just read, in seconds.

    ``CAP_PROP_POS_MSEC`` is preferred because it reflects the container's own
    timing, which matters for variable frame rate recordings. Some codecs
    report zero or a non-monotonic value, so the frame index is used as a
    fallback and the result is forced to increase.
    """
    position = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0

    unusable = (
        not math.isfinite(position)
        or position < 0
        or (frame_idx > 0 and position <= last_time)
    )

    if unusable:
        position = frame_idx / fps
        if frame_idx > 0:
            position = max(position, last_time + 1.0 / fps)

    return position


def write_droplet_videos(
    video_path: str,
    droplets: list[Droplet],
    out_dir: str,
    codec: str = "mp4v",
    fps: float | None = None,
    mask_background: bool = True,
    frame_callback: Callable[[int, float, "cv2.Mat"], None] | None = None,
    progress_every: int = 300,
) -> dict:
    """Export one ROI video per droplet.

    Parameters
    ----------
    video_path:
        The raw recording.
    droplets:
        Droplets from :func:`larvatracker.droplets.find_droplets`.
    out_dir:
        Directory for the ``droplet_XXX.mp4`` files; created if missing.
    codec:
        FourCC code handed to ``cv2.VideoWriter``.
    fps:
        Output frame rate. Defaults to the source frame rate, falling back to
        :data:`larvatracker.config.DEFAULT_FPS` if the container reports none.
    mask_background:
        Blank every pixel outside the droplet. Recommended: it prevents
        DeepLabCut from latching onto a larva in a neighbouring droplet that
        happens to overlap the bounding box.
    frame_callback:
        Called as ``callback(frame_idx, time_s, frame)`` for every decoded
        frame, before the ROIs are written. Used to read the temperature
        display in the same pass.
    progress_every:
        Print a progress line every N frames; set to 0 to silence.

    Returns
    -------
    dict
        ``out_dir``, ``frames`` and the ``fps`` actually used.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path!r}.")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not math.isfinite(fps) or fps <= 0:
        fps = DEFAULT_FPS

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writers: list[cv2.VideoWriter] = []

    try:
        for drop in droplets:
            path = os.path.join(out_dir, f"droplet_{drop.id:03d}.mp4")
            writer = cv2.VideoWriter(path, fourcc, fps, (drop.width, drop.height), True)

            if not writer.isOpened():
                raise RuntimeError(f"Could not open a video writer for {path!r}.")

            writers.append(writer)

        frame_idx = 0
        last_time = -1.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            time_s = frame_timestamp(cap, frame_idx, fps, last_time)
            last_time = time_s

            if frame_callback is not None:
                frame_callback(frame_idx, time_s, frame)

            for drop, writer in zip(droplets, writers):
                roi = frame[drop.y0:drop.y1, drop.x0:drop.x1].copy()

                if mask_background:
                    roi[~drop.roi_mask] = 0

                writer.write(roi)

            frame_idx += 1
            if progress_every and frame_idx % progress_every == 0:
                print(f"  frames written: {frame_idx}")
    finally:
        cap.release()
        for writer in writers:
            writer.release()

    print(f"Wrote {len(droplets)} ROI videos ({frame_idx} frames each) to {out_dir}")
    return {"out_dir": out_dir, "frames": frame_idx, "fps": float(fps)}
