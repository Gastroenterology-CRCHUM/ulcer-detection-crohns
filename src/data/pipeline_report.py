"""Shared utilities for preprocessing pipeline reports.

Provides per-stage directory stats (frames, clips, class counts, clip duration)
and annotation-window duration stats from Excel files.

Used by scripts/ulcer/eda.py, scripts/noninformative/eda.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _duration_stats(values: list[float]) -> dict:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "total": 0.0,
            "count": 0,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "total": float(arr.sum()),
        "count": len(arr),
    }


def _fmt_dur(stats: dict) -> str:
    if stats.get("count", 0) == 0:
        return "N/A"
    return (
        f"mean {stats['mean']:.1f}s ± {stats['std']:.1f}s"
        f"  [min {stats['min']:.1f} – max {stats['max']:.1f} | median {stats['median']:.1f}]"
        f"  total {stats['total'] / 60:.1f} min"
    )


# ---------------------------------------------------------------------------
# Directory-based stage stats
# ---------------------------------------------------------------------------


def collect_stage_stats(stage_dir: Path, fps: float) -> dict:
    """Count frames and clips per class in a 3-level directory structure.

    Structure: stage_dir / class_dir / video_dir / clip_dir / *.jpg

    Returns dict keyed by class name (e.g. "Ulcer", "score_0") plus "_total".
    Each value has: frames (int), clips (int), duration_s (stats dict).
    """
    if not stage_dir.exists():
        return {}

    result: dict[str, dict] = {}
    all_durations: list[float] = []

    for class_dir in sorted(stage_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        durations: list[float] = []
        total_frames = 0
        total_clips = 0

        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for clip_dir in sorted(video_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                n = sum(
                    1 for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                )
                if n == 0:
                    continue
                total_frames += n
                total_clips += 1
                durations.append(n / fps)

        if total_clips == 0:
            continue

        result[class_dir.name] = {
            "frames": total_frames,
            "clips": total_clips,
            "duration_s": _duration_stats(durations),
        }
        all_durations.extend(durations)

    if not result:
        return {}

    total_frames = sum(v["frames"] for v in result.values())
    total_clips = sum(v["clips"] for v in result.values())
    result["_total"] = {
        "frames": total_frames,
        "clips": total_clips,
        "duration_s": _duration_stats(all_durations),
    }
    return result


# ---------------------------------------------------------------------------
# Annotation-window duration from Excel
# ---------------------------------------------------------------------------


def annotation_duration_ulcer(excel_path: Path) -> dict | None:
    """Compute annotation-window duration stats from the ulcer Excel.

    Returns dict with per-label stats, or None on any error.
    """
    try:
        from src.data.annotation_loaders import load_ulcer_annotations

        df = load_ulcer_annotations(excel_path)
    except Exception as exc:
        logger.warning("Could not load ulcer annotations from %s: %s", excel_path, exc)
        return None

    df = df.copy()
    df["duration_s"] = df["end_s"] - df["start_s"]
    label_names = {1: "Ulcer", 0: "NonUlcer"}

    result: dict = {}
    for label, name in label_names.items():
        vals = df[df["label"] == label]["duration_s"].dropna().tolist()
        result[name] = _duration_stats(vals)
    result["_total"] = _duration_stats(df["duration_s"].dropna().tolist())
    result["_clips_per_label"] = {
        label_names.get(int(lbl), str(lbl)): int((df["label"] == lbl).sum())
        for lbl in sorted(df["label"].dropna().unique())
    }
    return result


# annotation_duration_mes removed — MES task not part of this project


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def format_pipeline_report(
    title: str,
    annotation_stats: dict | None,
    stages: list[tuple[str, dict]],
    fps: float,
    label_order: list[str] | None = None,
) -> str:
    """Format a text preprocessing pipeline report.

    Parameters
    ----------
    title         : Report title.
    annotation_stats : Output of annotation_duration_ulcer/mes (or None).
    stages        : List of (stage_name, stats_dict) from collect_stage_stats.
    fps           : Extraction FPS used for duration computation.
    label_order   : Class names in preferred display order.
    """
    W = 76
    lines: list[str] = []

    def h1(t: str) -> None:
        lines.extend(["=" * W, t.center(W), "=" * W])

    def h2(t: str) -> None:
        lines.extend([f"\n{t}", "-" * len(t)])

    h1(title)

    # ── Collect all class keys in order ───────────────────────────────────
    seen: list[str] = []
    for _, sd in stages:
        for k in sd:
            if k != "_total" and k not in seen:
                seen.append(k)
    if label_order:
        classes = [k for k in label_order if k in seen] + [k for k in seen if k not in label_order]
    else:
        classes = seen

    # ── 1. Annotation windows ─────────────────────────────────────────────
    if annotation_stats:
        h2("1. ANNOTATION WINDOWS (from Excel)")
        clips_per_label = annotation_stats.get("_clips_per_label", {})
        lines.append(f"  Total annotation windows : {sum(clips_per_label.values())}")
        for name, n in clips_per_label.items():
            lines.append(f"    {name:<20} {n:>5} windows")
        lines.append("")
        for name in clips_per_label:
            s = annotation_stats.get(name, {})
            if s.get("count", 0) > 0:
                lines.append(f"  {name:<20}  {_fmt_dur(s)}")
        s = annotation_stats.get("_total", {})
        if s.get("count", 0) > 0:
            lines.append(f"  {'Overall':<20}  {_fmt_dur(s)}")
    else:
        h2("1. ANNOTATION WINDOWS (from Excel)")
        lines.append("  Excel not available — skipped.")

    # ── 2. Frame and clip counts by stage ─────────────────────────────────
    h2("2. FRAME AND CLIP COUNTS BY STAGE")
    col_w = 9
    header = f"  {'Stage':<20}" + f"{'Frames':>{col_w}}" + f"{'Clips':>{col_w}}"
    for k in classes:
        short = k.replace("Non-Informative", "NonInf").replace("NonUlcer", "NonUlc")
        header += f"{short + ' fr':>{col_w}}"
    for k in classes:
        short = k.replace("Non-Informative", "NonInf").replace("NonUlcer", "NonUlc")
        header += f"{short + ' cl':>{col_w}}"
    lines.append(header)
    lines.append("  " + "-" * (20 + col_w * (2 + 2 * len(classes))))

    for stage_name, sd in stages:
        if not sd:
            lines.append(f"  {stage_name:<20}  (not found)")
            continue
        tot = sd.get("_total", {})
        row = f"  {stage_name:<20}{tot.get('frames', 0):>{col_w},}{tot.get('clips', 0):>{col_w},}"
        for k in classes:
            row += f"{sd.get(k, {}).get('frames', 0):>{col_w},}"
        for k in classes:
            row += f"{sd.get(k, {}).get('clips', 0):>{col_w},}"
        lines.append(row)

    # ── 3. Clip duration by stage ─────────────────────────────────────────
    h2(f"3. CLIP DURATION BY STAGE (@ {fps} FPS)")
    for stage_name, sd in stages:
        if not sd:
            continue
        lines.append(f"\n  {stage_name}")
        tot = sd.get("_total", {})
        lines.append(f"    {'Overall':<20}  {_fmt_dur(tot.get('duration_s', {}))}")
        for k in classes:
            s = sd.get(k, {}).get("duration_s", {})
            if s.get("count", 0) > 0:
                lines.append(f"    {k:<20}  {_fmt_dur(s)}")

    lines.append("\n" + "=" * W)
    return "\n".join(lines)
