"""Shared ROI extraction helpers for colonoscopy frames."""

from __future__ import annotations

import numpy as np


def crop_frac(img: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """Crop an image using fractional coordinates (y0, y1, x0, x1) in [0, 1]."""
    y0, y1, x0, x1 = roi
    H, W = img.shape[:2]
    return img[int(H * y0) : int(H * y1), int(W * x0) : int(W * x1)]
