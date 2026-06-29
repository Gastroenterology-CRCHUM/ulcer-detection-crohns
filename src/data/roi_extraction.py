"""Shared ROI extraction helpers for colonoscopy frames."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

ColorMode = Literal["bgr", "rgb"]


def crop_frac(img: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """Crop an image using fractional coordinates (y0, y1, x0, x1) in [0, 1]."""
    y0, y1, x0, x1 = roi
    H, W = img.shape[:2]
    return img[int(H * y0) : int(H * y1), int(W * x0) : int(W * x1)]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an image array to uint8."""
    if image.dtype == np.uint8:
        return image.copy()

    image_float = image.astype(np.float32)
    if image_float.size == 0:
        return image_float.astype(np.uint8)

    if float(np.nanmax(image_float)) <= 1.0:
        image_float = np.clip(image_float, 0.0, 1.0) * 255.0
    else:
        image_float = np.clip(image_float, 0.0, 255.0)

    return image_float.astype(np.uint8)


def build_roi_mask(
    image: np.ndarray,
    threshold: int = 100,
    *,
    color_mode: ColorMode = "bgr",
) -> np.ndarray:
    """Build a binary mask covering the visible octagonal field of view."""
    img_u8 = _to_uint8(image)
    amplified = np.clip(img_u8.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)

    if amplified.ndim == 3:
        if color_mode == "rgb":
            gray = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(amplified, cv2.COLOR_BGR2GRAY)
    else:
        gray = amplified

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(gray.shape, 255, dtype=np.uint8)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return mask


def extract_masked_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to an image."""
    img_u8 = _to_uint8(image)
    result = np.zeros_like(img_u8)
    result[mask > 0] = img_u8[mask > 0]
    return result


def crop_roi_frame(
    image: np.ndarray,
    threshold: int = 100,
    *,
    color_mode: ColorMode = "bgr",
    crop_left: int = 550,
    crop_right: int = 20,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply ROI masking and lateral crop."""
    mask = build_roi_mask(image, threshold=threshold, color_mode=color_mode)
    result = extract_masked_roi(image, mask)

    if result.ndim == 3 and result.shape[1] > crop_left + crop_right:
        result = result[:, crop_left : result.shape[1] - crop_right, :]

    if expected_shape is not None:
        assert result.shape[:2] == expected_shape, (
            f"Unexpected output shape {result.shape} — expected {expected_shape}"
        )

    return result
