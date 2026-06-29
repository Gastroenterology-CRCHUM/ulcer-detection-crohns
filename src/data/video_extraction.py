"""Generic video-to-frames extraction utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_video_index_generic(video_dir: Path) -> dict[str, Path]:
    """Map lowercased video stem → Path for all videos under video_dir."""
    idx: dict[str, Path] = {}
    for p in video_dir.rglob("*"):
        if p.is_file() and not p.name.startswith(("._", ".")) and p.suffix.lower() in VIDEO_EXTS:
            idx[p.stem.lower()] = p
    return idx


def sample_timestamps(start_s: float, end_s: float, fps_target: float) -> list[float]:
    """Generate timestamps spaced 1/fps_target apart between start_s and end_s."""
    if fps_target <= 0 or end_s <= start_s:
        return []
    step = 1.0 / fps_target
    return [float(v) for v in np.arange(start_s, end_s, step, dtype=float)]


def extract_frames_from_video(
    video_path: Path,
    start_s: float,
    end_s: float,
    out_dir: Path,
    *,
    fps_target: float = 1.0,
    offset_s: float = 0.0,
    jpeg_quality: int = 95,
    frame_prefix: str = "frame",
    skip_existing: bool = False,
) -> list[Path]:
    """Extract frames from [start_s, end_s] (overlay time) at fps_target.

    offset_s converts overlay time to real video time: video_ts = overlay_ts + offset_s.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    timestamps = sample_timestamps(start_s, end_s, fps_target)
    saved: list[Path] = []

    for ts in timestamps:
        video_ts = ts + offset_s
        if video_ts < 0 or (duration > 0 and video_ts >= duration):
            continue

        frame_number = int(round(video_ts * fps))
        dest = out_dir / f"{frame_prefix}_{frame_number:06d}.jpg"

        if skip_existing and dest.exists():
            saved.append(dest)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            continue

        if cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            saved.append(dest)
        else:
            logger.warning("imwrite failed: %s", dest)

    cap.release()
    return saved


def collect_frames_from_dir(frames_dir: Path) -> list[Path]:
    """Return all image files under frames_dir, sorted by path."""
    return sorted(
        p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
