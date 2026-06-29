"""
src/data/video_utils.py
-----------------------
Video and platform-detection utilities shared by the ulcer preprocessing pipeline.

Functions
---------
detect_green_rectangle(image)       → 'Fuji' | 'Olympus'
normalize_video_id(video_id)        → str
find_overlay_offset(video_path)     → float | None
_detect_platform_from_video(path)   → 'fuji' | 'olympus'
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError, image_to_string

from src.data.roi_extraction import crop_frac

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEFT_PANEL_ROI = (0.00, 1.00, 0.00, 0.32)  # (y0, y1, x0, x1)
_TESSERACT_WARNED = False
_TESSERACT_RESOLVED = False


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_green_rectangle(image: np.ndarray) -> str:
    """
    Detect equipment type (Fuji or Olympus) from the upper-right corner ROI.

    Args:
        image: Input frame in BGR format.

    Returns:
        'Fuji' or 'Olympus'.
    """
    height, width, _ = image.shape
    square_size = min(height, width) // 20
    roi = image[0:square_size, width - square_size: width]

    mean_blue = np.mean(roi[:, :, 0])
    mean_green = np.mean(roi[:, :, 1])
    mean_red = np.mean(roi[:, :, 2])
    mean_total = np.mean(roi)

    if mean_total < 1.0:
        return "Olympus"
    elif mean_green > 20.0 and mean_red < 10.0:
        return "Fuji"
    elif 12.0 <= mean_total <= 16.0 and abs(mean_blue - mean_red) < 1.0:
        return "Olympus"
    else:
        return "Fuji" if mean_green > 25.0 else "Olympus"


def _detect_platform_from_video(video_path: Path) -> str:
    """Return 'fuji' or 'olympus' by inspecting the first readable frame."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    platform = "olympus"
    for seek_s in (60, 30, 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_s * fps))
        ok, frame = cap.read()
        if ok:
            platform = detect_green_rectangle(frame).lower()
            break
    cap.release()
    return platform


# ---------------------------------------------------------------------------
# Video ID normalisation
# ---------------------------------------------------------------------------


def normalize_video_id(video_id: str) -> str:
    rid = str(video_id).strip().lower()
    rid = rid.replace("-", "_")
    rid = re.sub(r"\s+", "", rid)
    return rid


# ---------------------------------------------------------------------------
# OCR overlay-offset helpers
# ---------------------------------------------------------------------------


def _normalize_ocr_text(text: str) -> str:
    text = str(text).upper()
    text = text.replace("O", "0").replace("S", "5").replace("I", "1").replace("L", "1")
    return re.sub(r"[^0-9:\-\n ]", " ", text)


def _parse_stopwatch_seconds(
    text: str, *, is_fuji: bool = False, hint_sec: float | None = None
) -> int | None:
    """Parse stopwatch text from Fuji or Olympus overlays."""
    cleaned = _normalize_ocr_text(text)

    if is_fuji:
        matches = re.findall(r"(?<!\d:)\b(\d{1,3})\s*:\s*([0-5]\d)\b(?!:\d)", cleaned)
        if not matches:
            return None
        values = [int(m) * 60 + int(s) for m, s in matches]
        if hint_sec is not None:
            target = float(hint_sec) % 3600.0
            return int(min(values, key=lambda v: abs((float(v) % 3600.0) - target)))
        return int(values[-1])

    matches = re.findall(r"\b(\d{2}):([0-5]\d):([0-5]\d)\b", cleaned)
    if matches:
        if hint_sec is not None:
            target = float(hint_sec) % 3600.0
            best_value = None
            best_distance = float("inf")
            for h, m, s in matches:
                value = int(h) * 3600 + int(m) * 60 + int(s)
                distance = abs((float(value) % 3600.0) - target)
                if distance < best_distance:
                    best_value = value
                    best_distance = distance
            return best_value
        for h, m, s in matches:
            if h == "00":
                return int(h) * 3600 + int(m) * 60 + int(s)
        h, m, s = matches[-1]
        return int(h) * 3600 + int(m) * 60 + int(s)

    fuji_matches = re.findall(r"(?<!\d:)\b(\d{1,3})\s*:\s*([0-5]\d)\b(?!:\d)", cleaned)
    if not fuji_matches:
        return None
    fuji_values = [int(m) * 60 + int(s) for m, s in fuji_matches]
    if hint_sec is not None:
        target = float(hint_sec) % 3600.0
        return int(min(fuji_values, key=lambda v: abs((float(v) % 3600.0) - target)))
    return int(fuji_values[-1])


def _ensure_tesseract_path() -> bool:
    global _TESSERACT_RESOLVED
    if _TESSERACT_RESOLVED:
        return True
    if shutil.which("tesseract"):
        _TESSERACT_RESOLVED = True
        return True
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            _TESSERACT_RESOLVED = True
            return True
    return False


def ocr_left_panel_text(frame_bgr: np.ndarray, save_debug: str | None = None) -> str:
    global _TESSERACT_WARNED
    _ensure_tesseract_path()

    roi = crop_frac(frame_bgr, LEFT_PANEL_ROI)
    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
    scale = 1.6
    gray = cv2.resize(
        gray,
        (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
        interpolation=cv2.INTER_CUBIC,
    )

    candidates = []
    for thflag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, thr = cv2.threshold(gray, 0, 255, thflag | cv2.THRESH_OTSU)
        if save_debug and thflag == cv2.THRESH_BINARY_INV:
            cv2.imwrite(save_debug, thr)
        try:
            txt = image_to_string(
                thr, config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:"
            )
        except TesseractNotFoundError:
            if not _TESSERACT_WARNED:
                print(
                    "WARNING: Tesseract is not installed or not in PATH. "
                    "OCR offset detection will be unavailable until Tesseract is installed."
                )
                _TESSERACT_WARNED = True
            return ""
        candidates.append(txt)

    return _normalize_ocr_text("\n".join(candidates))


def find_overlay_offset(
    video_path: Union[str, Path],
    probe_times: tuple[int, ...] = (120, 180, 240),
    debug_dir: Union[str, Path] | None = "ocr_debug/ulcer",
    is_fuji: bool = False,
) -> float | None:
    """Return offset = video_time - overlay_time, estimated from OCR probes."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0

    probe_queue: list[int] = list(probe_times)
    if duration > 0:
        for t in range(60, int(min(duration, 600)) + 1, 30):
            if t not in probe_queue:
                probe_queue.append(t)

    for t in probe_queue:
        if duration > 0 and t >= duration:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue

        dbg_path = None
        if debug_dir:
            dbg_root = Path(debug_dir)
            dbg_root.mkdir(parents=True, exist_ok=True)
            dbg_path = str(dbg_root / f"{video_path.stem}_leftpanel_t{t}.png")

        txt = ocr_left_panel_text(frame, save_debug=dbg_path)
        overlay_sec = _parse_stopwatch_seconds(txt, is_fuji=is_fuji, hint_sec=t)
        if overlay_sec is not None:
            print(
                f"[{video_path.name}] OCR probe t={t}s → overlay={overlay_sec}s "
                f"→ offset={t - overlay_sec:+.2f}s"
            )
            cap.release()
            return float(t) - float(overlay_sec)

    cap.release()
    print(f"[{video_path.name}] OCR couldn't read stopwatch at probes {probe_queue}.")
    return None
