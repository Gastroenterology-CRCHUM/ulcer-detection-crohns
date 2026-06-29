"""Shared ROI preprocessing — platform-aware crop and octagonal mask.

Handles both Fuji and Olympus colonoscopy platforms. Builds ONE strict
octagonal reference mask from an Olympus frame and applies it to all frames.

Input : any raw frame directory (1920×1080 JPEG)
Output: cropped and masked frames (1080×1350 JPEG)

Usage
-----
    python scripts/data/preprocess_frames.py \\
        --raw-dir data/ulcer/raw \\
        --output-dir data/ulcer/processed

    python scripts/data/preprocess_frames.py \\
        --raw-dir data/mes/raw \\
        --output-dir data/mes/processed \\
        --incremental
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.data.video_utils import detect_green_rectangle, normalize_video_id

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_THRESHOLDS = [20, 30, 40, 55, 70, 90, 110]
TARGET_HW = (1080, 1350)  # (height, width)

# Values derived from notebooks/crop_helper.ipynb
OLYMPUS_CROP = {"y1": 0, "y2": -0, "x1": 550, "x2_right": 20}
FUJI_CROP = {"y1": 60, "y2_bottom": 60, "x1": 140, "x2_right": 690}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROI-crop and octagonal-mask raw frames (Fuji/Olympus)."
    )
    parser.add_argument("--raw-dir", type=str, required=True, help="Root directory of raw frames.")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to write processed frames."
    )
    parser.add_argument(
        "--olympus-mask-path",
        type=str,
        default=None,
        help="Path to save the Olympus reference mask (optional, for verification).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Resize output to a square of this size after crop/mask (e.g. 224). "
             "Saves disk space and speeds up DataLoader when the model input "
             "is smaller than TARGET_HW. No effect if None (default).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip frames already present in output-dir.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Skip ROI crop and octagonal masking — just copy/resize frames. "
             "Use for datasets without endoscope border (e.g. LIMUC).",
    )
    parser.add_argument(
        "--convert-bmp",
        action="store_true",
        help="Convert BMP source files to PNG at the destination instead of keeping BMP.",
    )
    return parser


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _crop_platform(frame: np.ndarray, platform: str) -> np.ndarray:
    h, w = frame.shape[:2]
    if platform == "fuji":
        y1 = min(max(FUJI_CROP["y1"], 0), h)
        y2 = max(y1 + 1, h - FUJI_CROP["y2_bottom"])
        x1 = min(max(FUJI_CROP["x1"], 0), w)
        x2 = max(x1 + 1, w - FUJI_CROP["x2_right"])
        return frame[y1:y2, x1:x2]
    y1 = min(max(OLYMPUS_CROP["y1"], 0), h)
    y2 = h
    x1 = min(max(OLYMPUS_CROP["x1"], 0), w)
    x2 = max(x1 + 1, w - OLYMPUS_CROP["x2_right"])
    return frame[y1:y2, x1:x2]


def _build_platform_map(image_paths: list[Path], raw_dir: Path) -> dict[str, str]:
    """Detect Fuji vs Olympus from a representative frame for each video_id (parts[1])."""
    vid_to_first: dict[str, Path] = {}
    for path in image_paths:
        rel = path.relative_to(raw_dir)
        if len(rel.parts) < 2:
            continue
        vid = normalize_video_id(rel.parts[1])
        if vid not in vid_to_first:
            vid_to_first[vid] = path

    platform_map: dict[str, str] = {}
    for vid, path in tqdm(vid_to_first.items(), desc="Detect platforms", unit="video"):
        frame = cv2.imread(str(path))
        if frame is None:
            platform_map[vid] = "olympus"
            continue
        platform_map[vid] = detect_green_rectangle(frame).lower()
    return platform_map


def _has_overlay_artifacts(cropped: np.ndarray) -> bool:
    if cropped.size == 0:
        return True
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    h, _ = gray.shape
    top_h = max(24, int(0.14 * h))
    top_band = gray[:top_h, :]
    edges = cv2.Canny(top_band, 120, 220)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    small_components = sum(
        1 for i in range(1, n_labels) if 4 <= int(stats[i, cv2.CC_STAT_AREA]) <= 120
    )
    edge_density = float((edges > 0).mean())
    return small_components > 70 or edge_density > 0.07


def _strict_octagon_mask(
    cropped: np.ndarray, threshold: int
) -> tuple[np.ndarray | None, int, float]:
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    h, w = gray.shape
    area_ratio = area / max(float(h * w), 1.0)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = int(len(approx))
    if vertices != 8:
        return None, vertices, area_ratio
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillPoly(mask, [approx], 255)
    return mask, vertices, area_ratio


def _build_shared_olympus_mask(
    image_paths: list[Path],
    raw_dir: Path,
    platform_map: dict[str, str],
) -> np.ndarray:
    olympus_paths = [
        p
        for p in image_paths
        if len(p.relative_to(raw_dir).parts) >= 2
        and platform_map.get(normalize_video_id(p.relative_to(raw_dir).parts[1]), "olympus")
        != "fuji"
    ]
    if not olympus_paths:
        raise RuntimeError("No Olympus frames found to build a shared reference mask.")

    step = max(1, len(olympus_paths) // 700)
    best_score = -1e9
    best_mask = None
    best_info = None

    for src_path in tqdm(olympus_paths[::step], desc="Build shared Olympus mask", unit="img"):
        frame = cv2.imread(str(src_path))
        if frame is None:
            continue
        cropped = _crop_platform(frame, "olympus")
        if cropped.size == 0 or _has_overlay_artifacts(cropped):
            continue
        for th in MASK_THRESHOLDS:
            mask, vertices, area_ratio = _strict_octagon_mask(cropped, threshold=th)
            score = (2.0 if vertices == 8 else 0.0) + area_ratio - 0.12 * abs(vertices - 8)
            if score > best_score and mask is not None:
                best_score = score
                best_mask = mask
                best_info = (src_path.name, th, area_ratio, vertices)

    if best_mask is None:
        frame = cv2.imread(str(olympus_paths[0]))
        if frame is None:
            raise RuntimeError("Cannot read Olympus reference frame for fallback mask.")
        cropped = _crop_platform(frame, "olympus")
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, best_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        best_info = (olympus_paths[0].name, 1, float((best_mask > 0).mean()), -1)

    ref_name, ref_th, ref_area, ref_vertices = best_info  # type: ignore[misc]
    print(
        f"Shared Olympus mask from {ref_name} | threshold={ref_th} "
        f"| area_ratio={ref_area:.3f} | vertices={ref_vertices}"
    )
    return best_mask


def preprocess_frames(
    raw_dir: Path,
    output_dir: Path,
    jpeg_quality: int = 95,
    incremental: bool = False,
    olympus_mask_path: Path | None = None,
    default_platform: str | None = None,
    skip_crop: bool = False,
    convert_bmp: bool = False,
    target_size: int | None = None,
) -> dict:
    """ROI-crop and octagonally mask raw colonoscopy frames.

    Parameters
    ----------
    default_platform:
        Force a single platform ('olympus' or 'fuji') for all frames,
        skipping per-video detection. Useful for flat directory structures
        (e.g. the informative pipeline) where the directory layout does not
        carry video-level metadata.
    skip_crop:
        Skip ROI crop and octagonal masking entirely — frames are only
        resized to TARGET_HW.  Use for datasets without endoscope border
        (e.g. LIMUC).
    convert_bmp:
        When the source file is a BMP, write the output as PNG instead of
        keeping the BMP extension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _iter_images(raw_dir)
    if not image_paths:
        raise RuntimeError(f"No raw frames found under: {raw_dir}")

    target_h, target_w = TARGET_HW

    if not skip_crop:
        # Platform detection — skip when a default is forced
        platform_map: dict[str, str] = (
            {} if default_platform is not None else _build_platform_map(image_paths, raw_dir)
        )

        shared_mask = _build_shared_olympus_mask(image_paths, raw_dir, platform_map)

        if shared_mask.shape[:2] != TARGET_HW:
            shared_mask = cv2.resize(shared_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        if olympus_mask_path is not None:
            olympus_mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(olympus_mask_path), shared_mask)
    else:
        platform_map = {}
        shared_mask = None

    n_ok = n_failed = n_skipped = 0
    platform_counts: dict[str, int] = {}

    for src_path in tqdm(image_paths, desc="ROI preprocess", unit="img"):
        rel_path = src_path.relative_to(raw_dir)
        dst_path = output_dir / rel_path

        # BMP → PNG conversion at destination
        if convert_bmp and src_path.suffix.lower() == ".bmp":
            dst_path = dst_path.with_suffix(".png")

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if incremental and dst_path.exists():
            n_skipped += 1
            continue

        frame = cv2.imread(str(src_path))
        if frame is None:
            n_failed += 1
            continue

        if skip_crop:
            # No crop or mask — just resize
            processed = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            if default_platform is not None:
                platform = default_platform
            else:
                parts = rel_path.parts
                vid = normalize_video_id(parts[1]) if len(parts) >= 2 else ""
                platform = platform_map.get(vid, "olympus")

            platform_counts[platform] = platform_counts.get(platform, 0) + 1

            processed = _crop_platform(frame, platform)
            if processed.size == 0:
                n_failed += 1
                continue

            if processed.shape[:2] != TARGET_HW:
                processed = cv2.resize(processed, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            mask = shared_mask
            if mask.shape[:2] != processed.shape[:2]:
                mask = cv2.resize(
                    mask, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            processed = cv2.bitwise_and(processed, processed, mask=mask)

        if target_size is not None and processed.shape[:2] != (target_size, target_size):
            processed = cv2.resize(processed, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        if dst_path.suffix.lower() == ".png":
            ok = cv2.imwrite(str(dst_path), processed)
        else:
            ok = cv2.imwrite(str(dst_path), processed, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            n_ok += 1
        else:
            n_failed += 1

    stats = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(output_dir),
        "total_input_frames": len(image_paths),
        "preprocessed_frames": n_ok,
        "skipped_frames": n_skipped,
        "failed_frames": n_failed,
        "platform_counts": platform_counts,
        "olympus_mask_path": str(olympus_mask_path) if olympus_mask_path else None,
    }
    with open(output_dir / "preprocess_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    return stats


def main(args: argparse.Namespace) -> None:
    stats = preprocess_frames(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        jpeg_quality=args.jpeg_quality,
        incremental=getattr(args, "incremental", False),
        olympus_mask_path=Path(args.olympus_mask_path) if args.olympus_mask_path else None,
        skip_crop=getattr(args, "no_crop", False),
        convert_bmp=getattr(args, "convert_bmp", False),
        target_size=getattr(args, "target_size", None),
    )

    print("=" * 72)
    print("ROI PREPROCESS DONE")
    print("=" * 72)
    print(f"Input frames  : {stats['total_input_frames']}")
    print(f"Preprocessed  : {stats['preprocessed_frames']}")
    print(f"Skipped       : {stats['skipped_frames']}")
    print(f"Failed        : {stats['failed_frames']}")
    print(f"Output dir    : {stats['processed_dir']}")
    print("=" * 72)


if __name__ == "__main__":
    main(build_parser().parse_args())
