"""
tests/test_roi_extraction.py
============================
Unit tests for shared ROI extraction helpers used by scripts.
"""

import cv2
import numpy as np

from src.data.roi_extraction import build_roi_mask, crop_roi_frame, extract_masked_roi


def _make_raw_frame(height: int = 1080, width: int = 1920) -> np.ndarray:
    """Create a synthetic frame with a bright circular field of view."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(height, width) // 3
    cv2.circle(frame, center, radius, (255, 255, 255), thickness=-1)
    return frame


def test_build_roi_mask_returns_binary_mask():
    frame = _make_raw_frame()
    mask = build_roi_mask(frame, color_mode="bgr")

    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask.min() in (0, 255)
    assert mask.sum() > 0


def test_extract_masked_roi_preserves_shape():
    frame = _make_raw_frame()
    mask = build_roi_mask(frame, color_mode="bgr")
    roi = extract_masked_roi(frame, mask)

    assert roi.shape == frame.shape
    assert roi.dtype == np.uint8
    assert roi.sum() > 0


def test_crop_roi_frame_matches_expected_ulcer_shape():
    frame = _make_raw_frame()
    cropped = crop_roi_frame(
        frame,
        color_mode="bgr",
        crop_left=550,
        crop_right=20,
        expected_shape=(1080, 1350),
    )

    assert cropped.shape == (1080, 1350, 3)


def test_crop_roi_frame_supports_rgb_mode():
    frame = _make_raw_frame()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cropped = crop_roi_frame(
        rgb_frame,
        color_mode="rgb",
        crop_left=550,
        crop_right=20,
        expected_shape=(1080, 1350),
    )

    assert cropped.shape == (1080, 1350, 3)
