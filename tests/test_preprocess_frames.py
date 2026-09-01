"""Tests for scripts/data/preprocess_frames.py, focused on the already-cropped
input detection: a frame whose dimensions don't match the raw 1920x1080
canonical size must not be re-cropped with the fixed platform pixel offsets.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.data.preprocess_frames import TARGET_HW, _build_platform_map, preprocess_frames
from src.data.video_utils import RAW_HW, is_raw_shaped


def _write(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def _raw_frame_with_corner_marker(marker: str | None = None) -> np.ndarray:
    """A raw-shaped (1920x1080) frame; an all-black corner reads as Olympus."""
    h, w = RAW_HW
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if marker == "fuji":
        square = min(h, w) // 20
        frame[0:square, w - square : w] = (0, 255, 0)  # BGR pure green -> Fuji
    return frame


# ---------------------------------------------------------------------------
# is_raw_shaped
# ---------------------------------------------------------------------------


def test_is_raw_shaped_true_for_canonical_resolution():
    frame = np.zeros((*RAW_HW, 3), dtype=np.uint8)
    assert is_raw_shaped(frame) is True


@pytest.mark.parametrize("shape", [(960, 1090), (1080, 1350), (720, 1280)])
def test_is_raw_shaped_false_for_other_resolutions(shape):
    frame = np.zeros((*shape, 3), dtype=np.uint8)
    assert is_raw_shaped(frame) is False


# ---------------------------------------------------------------------------
# _build_platform_map
# ---------------------------------------------------------------------------


def test_build_platform_map_skips_precropped_candidate(tmp_path):
    raw_dir = tmp_path
    vid_dir = raw_dir / "Ulcer" / "vid_01" / "seg_1"
    # First candidate already looks cropped, must be skipped for detection:
    # its corner marker (if any) was already cut off by a previous crop.
    _write(vid_dir / "a_frame.jpg", np.zeros((960, 1090, 3), dtype=np.uint8))
    # Second candidate is raw-shaped with a genuine Fuji corner marker.
    _write(vid_dir / "b_frame.jpg", _raw_frame_with_corner_marker("fuji"))

    image_paths = sorted(vid_dir.glob("*.jpg"))
    platform_map = _build_platform_map(image_paths, raw_dir)
    assert platform_map["vid_01"] == "fuji"


def test_build_platform_map_defaults_to_olympus_when_all_precropped(tmp_path):
    raw_dir = tmp_path
    vid_dir = raw_dir / "Ulcer" / "vid_02" / "seg_1"
    _write(vid_dir / "a_frame.jpg", np.zeros((960, 1090, 3), dtype=np.uint8))

    image_paths = sorted(vid_dir.glob("*.jpg"))
    platform_map = _build_platform_map(image_paths, raw_dir)
    assert platform_map["vid_02"] == "olympus"


# ---------------------------------------------------------------------------
# preprocess_frames — mixed raw / already-cropped input
# ---------------------------------------------------------------------------


def test_preprocess_frames_skips_recrop_for_already_cropped_input(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"

    # A genuine raw Olympus frame, with bright content inside the kept crop
    # region (x >= 550) so the fallback mask-builder has something to find.
    raw_frame = _raw_frame_with_corner_marker()
    raw_frame[100:900, 700:1800] = 200
    _write(raw_dir / "Ulcer" / "vid_a" / "seg_1" / "frame_000.jpg", raw_frame)

    # A frame that is already ROI-cropped (Fuji post-crop, pre-resize size) -
    # applying crop_platform's fixed offsets to it again would slice garbage.
    precropped_frame = np.full((960, 1090, 3), 150, dtype=np.uint8)
    _write(raw_dir / "Ulcer" / "vid_b" / "seg_1" / "frame_000.jpg", precropped_frame)

    stats = preprocess_frames(raw_dir=raw_dir, output_dir=out_dir)

    assert stats["already_cropped_frames"] == 1
    assert stats["preprocessed_frames"] == 2
    assert stats["failed_frames"] == 0

    out_a = out_dir / "Ulcer" / "vid_a" / "seg_1" / "frame_000.jpg"
    out_b = out_dir / "Ulcer" / "vid_b" / "seg_1" / "frame_000.jpg"
    assert cv2.imread(str(out_a)).shape[:2] == TARGET_HW
    assert cv2.imread(str(out_b)).shape[:2] == TARGET_HW
