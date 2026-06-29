"""Temporal interval sampling utilities for video annotation pipelines."""

from __future__ import annotations

import random


def sample_intervals_random(
    total_frames: int,
    fps: float,
    interval_sec: float,
    n_intervals: int,
    seed: int | None = None,
) -> list[tuple[int, int]]:
    """Pick non-overlapping random intervals of *interval_sec* seconds.

    Args:
        total_frames: Total number of frames in the video.
        fps:          Video frame rate.
        interval_sec: Duration of each interval in seconds.
        n_intervals:  Maximum number of intervals to sample.
        seed:         Random seed for reproducibility.

    Returns:
        Sorted list of ``(start_frame, end_frame)`` pairs.
        May be shorter than *n_intervals* if the video is too short.
    """
    rng = random.Random(seed)
    interval_frames = int(interval_sec * fps)

    all_starts = list(range(0, total_frames - interval_frames, interval_frames))
    if not all_starts:
        return []

    chosen = rng.sample(all_starts, min(n_intervals, len(all_starts)))
    return sorted((s, s + interval_frames) for s in chosen)


def build_frame_queue(
    cap,
    fps: float,
    sample_fps: float,
    intervals: list[tuple],
) -> list[dict]:
    """Convert a list of frame intervals into an ordered annotation queue.

    Args:
        cap:        ``cv2.VideoCapture`` (used only to validate — not read here).
        fps:        Native video frame rate.
        sample_fps: Desired extraction frame rate (≤ fps).
        intervals:  List of ``(start_frame, end_frame)`` or
                    ``(start_frame, end_frame, hint)`` tuples.

    Returns:
        List of dicts with keys:
            ``frame_idx``   – absolute frame index in the video,
            ``timestamp_s`` – position in seconds,
            ``interval_idx``– which interval this frame belongs to,
            ``hint``        – category hint string (empty if not provided).
    """
    step = max(1, int(fps / sample_fps))
    queue = []
    for i, interval in enumerate(intervals):
        sf, ef = interval[0], interval[1]
        hint = interval[2] if len(interval) > 2 else ""
        for f in range(sf, ef, step):
            queue.append(
                {
                    "frame_idx": f,
                    "timestamp_s": f / fps,
                    "interval_idx": i,
                    "hint": hint,
                }
            )
    return queue
