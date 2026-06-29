"""Shared EDA utilities for ulcer and MES preprocessing pipelines.

Generic plotting and analysis functions used by both
scripts/ulcer/eda.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    return obj


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Frame-flow and filtering
# ---------------------------------------------------------------------------


def plot_frame_flow(
    raw_count: int,
    processed_count: int,
    filtrated_count: int,
    output_dir: Path,
    title: str = "Frame Flow",
) -> None:
    sns.set_palette("Set2")
    names = ["Raw", "Processed", "Filtrated"]
    values = [raw_count, processed_count, filtrated_count]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, edgecolor="black")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{value:,}",
            ha="center",
        )
    ax.set_title(title)
    ax.set_ylabel("Number of frames")
    plt.tight_layout()
    plt.savefig(output_dir / "frame_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_filter_outcomes(pred_df: pd.DataFrame, output_dir: Path) -> None:
    if pred_df.empty or "category" not in pred_df.columns:
        return
    counts = (
        pred_df["category"]
        .value_counts()
        .reindex(["informative", "non_informative", "uncertain"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index, counts.values, color=["#2ecc71", "#e74c3c", "#f39c12"], edgecolor="black"
    )
    total = max(counts.sum(), 1)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            f"{value:,} ({100.0 * value / total:.1f}%)",
            ha="center",
        )
    ax.set_title("Filtering outcomes")
    ax.set_ylabel("Frames")
    plt.tight_layout()
    plt.savefig(output_dir / "filter_outcomes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_video_retention(pred_df: pd.DataFrame, output_dir: Path) -> None:
    if pred_df.empty or "video_id" not in pred_df.columns:
        return
    summary = pred_df.groupby("video_id")["category"].value_counts().unstack(fill_value=0)
    for col in ("informative", "non_informative", "uncertain"):
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[["informative", "non_informative", "uncertain"]].copy()
    summary["total"] = summary.sum(axis=1)
    summary["kept_ratio"] = summary["informative"] / summary["total"].clip(lower=1)
    summary = summary.sort_values("kept_ratio", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary.index, summary["kept_ratio"].values, color="#2ecc71", edgecolor="black")
    ax.set_title("Per-video informative retention ratio")
    ax.set_ylabel("Kept ratio")
    ax.set_xlabel("Video")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "video_retention_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest-level plots
# ---------------------------------------------------------------------------


def plot_manifest_distributions(
    manifest: pd.DataFrame, output_dir: Path, label_xlabel: str = "Label"
) -> None:
    if manifest.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    label_counts = manifest["label"].value_counts().sort_index()
    bars = axes[0].bar([str(i) for i in label_counts.index], label_counts.values, edgecolor="black")
    for bar, value in zip(bars, label_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(label_counts.values) * 0.01,
            f"{value:,}",
            ha="center",
        )
    axes[0].set_title("Manifest label distribution")
    axes[0].set_xlabel(label_xlabel)
    axes[0].set_ylabel("Frames")

    if "split" in manifest.columns:
        split_counts = (
            manifest["split"].value_counts().reindex(["train", "val", "test"], fill_value=0)
        )
        bars = axes[1].bar(split_counts.index, split_counts.values, edgecolor="black")
        for bar, value in zip(bars, split_counts.values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(split_counts.values) * 0.01,
                f"{value:,}",
                ha="center",
            )
        axes[1].set_title("Split distribution")
        axes[1].set_ylabel("Frames")
    else:
        axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "manifest_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_label_distribution(manifest: pd.DataFrame, output_dir: Path) -> None:
    if manifest.empty or "split" not in manifest.columns or "label" not in manifest.columns:
        return
    split_order = ["train", "val", "test"]
    label_order = sorted(manifest["label"].dropna().unique())
    if not label_order:
        return
    table = (
        manifest.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=split_order, fill_value=0)
        .reindex(columns=label_order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(table.index), dtype=float)
    x = np.arange(len(table.index))
    palette = sns.color_palette("tab10", n_colors=len(label_order))
    for idx, label in enumerate(label_order):
        values = table[label].to_numpy(dtype=float)
        bars = ax.bar(
            x, values, bottom=bottom, color=palette[idx], edgecolor="black", label=f"label {label}"
        )
        for bar, v, b in zip(bars, values, bottom):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + v / 2,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylabel("Frames")
    ax.set_title("Label distribution by split")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "split_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frames_per_video(manifest: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    if manifest.empty or "video_id" not in manifest.columns:
        return
    counts = manifest["video_id"].value_counts().head(top_n)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(counts.index.astype(str), counts.values, color="#3498db", edgecolor="black")
    ax.set_title(f"Top {top_n} videos by frame count")
    ax.set_ylabel("Frames")
    ax.set_xlabel("Video ID")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "top_videos_frame_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frames_per_clip(manifest: pd.DataFrame, output_dir: Path) -> None:
    if manifest.empty:
        return
    clip_col = "clip_key" if "clip_key" in manifest.columns else None
    if clip_col is None and {"video_id", "segment_id"}.issubset(set(manifest.columns)):
        clip_col = "_clip_key"
        manifest = manifest.copy()
        manifest[clip_col] = (
            manifest["video_id"].astype(str) + "__" + manifest["segment_id"].astype(str)
        )
    if clip_col is None:
        return
    clip_sizes = manifest.groupby(clip_col).size()
    if clip_sizes.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(clip_sizes.values, bins=30, color="#9b59b6", edgecolor="black")
    ax.set_title("Frames per clip distribution")
    ax.set_xlabel("Frames per clip")
    ax.set_ylabel("Number of clips")
    plt.tight_layout()
    plt.savefig(output_dir / "frames_per_clip_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frame_number_by_label(manifest: pd.DataFrame, output_dir: Path) -> None:
    if manifest.empty or "frame_number" not in manifest.columns or "label" not in manifest.columns:
        return
    plot_df = manifest[["label", "frame_number"]].dropna().copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=plot_df, x="label", y="frame_number", ax=ax)
    ax.set_title("Frame number distribution by label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Frame number")
    plt.tight_layout()
    plt.savefig(output_dir / "frame_number_by_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest quality and split diagnostics
# ---------------------------------------------------------------------------


def manifest_quality_checks(manifest: pd.DataFrame) -> dict:
    if manifest.empty:
        return {"rows": 0, "missing_values_per_column": {}, "duplicate_relative_path": 0}
    dup_rel = (
        int(manifest["relative_path"].duplicated().sum())
        if "relative_path" in manifest.columns
        else 0
    )
    clip_frame_cols = [
        c for c in ["video_id", "segment_id", "frame_number"] if c in manifest.columns
    ]
    dup_clip_frame = (
        int(manifest.duplicated(subset=clip_frame_cols).sum()) if len(clip_frame_cols) == 3 else 0
    )
    expected_cols = [
        "relative_path",
        "label",
        "video_id",
        "patient_id",
        "segment_id",
        "frame_number",
        "clip_key",
        "split",
    ]
    return {
        "rows": int(len(manifest)),
        "missing_values_per_column": {
            str(col): int(val) for col, val in manifest.isna().sum().items()
        },
        "duplicate_relative_path": dup_rel,
        "duplicate_clip_frame": dup_clip_frame,
        "expected_columns_present": {col: col in manifest.columns for col in expected_cols},
    }


def split_diagnostics(manifest: pd.DataFrame) -> dict:
    if manifest.empty or "split" not in manifest.columns:
        return {
            "split_counts": {},
            "label_by_split": {},
            "patient_leakage": {},
            "label_shift_vs_global": {},
        }
    split_counts = {str(k): int(v) for k, v in manifest["split"].value_counts().items()}
    label_by_split = {}
    label_shift: dict[str, float] = {}
    if "label" in manifest.columns:
        ctab = pd.crosstab(manifest["split"], manifest["label"], normalize="index")
        label_by_split = {
            str(s): {str(lbl): float(v) for lbl, v in row.items()} for s, row in ctab.iterrows()
        }
        global_dist = manifest["label"].value_counts(normalize=True).sort_index()
        for split_name, split_df in manifest.groupby("split"):
            split_dist = split_df["label"].value_counts(normalize=True).sort_index()
            all_labels = sorted(set(global_dist.index).union(set(split_dist.index)))
            label_shift[str(split_name)] = round(
                max(
                    abs(float(global_dist.get(lbl, 0)) - float(split_dist.get(lbl, 0)))
                    for lbl in all_labels
                ),
                4,
            )
    patient_leakage = {}
    if "patient_id" in manifest.columns:
        sp = {
            str(s): set(df["patient_id"].dropna().astype(str).unique())
            for s, df in manifest.groupby("split")
        }
        names = sorted(sp)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap = sp[a].intersection(sp[b])
                patient_leakage[f"{a}_vs_{b}"] = {
                    "count": int(len(overlap)),
                    "examples": sorted(list(overlap))[:10],
                }
    return {
        "split_counts": split_counts,
        "label_by_split": label_by_split,
        "patient_leakage": patient_leakage,
        "label_shift_vs_global": label_shift,
    }


def entity_summaries(manifest: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if manifest.empty:
        return (
            {
                "videos": 0,
                "patients": 0,
                "segments": 0,
                "clips": 0,
                "frames_per_video_top15": {},
                "frames_per_split": {},
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )
    videos = int(manifest["video_id"].nunique()) if "video_id" in manifest.columns else 0
    patients = int(manifest["patient_id"].nunique()) if "patient_id" in manifest.columns else 0
    segments = int(manifest["segment_id"].nunique()) if "segment_id" in manifest.columns else 0
    clips = int(manifest["clip_key"].nunique()) if "clip_key" in manifest.columns else 0

    video_summary = pd.DataFrame()
    if "video_id" in manifest.columns:
        agg_spec: dict = {"frames": ("video_id", "count")}
        if "label" in manifest.columns:
            agg_spec["n_labels"] = ("label", "nunique")
        if "segment_id" in manifest.columns:
            agg_spec["n_segments"] = ("segment_id", "nunique")
        if "split" in manifest.columns:
            agg_spec["split"] = ("split", lambda x: ",".join(sorted(set(x.astype(str)))))
        video_summary = (
            manifest.groupby("video_id")
            .agg(**agg_spec)
            .reset_index()
            .sort_values("frames", ascending=False)
        )

    clip_summary = pd.DataFrame()
    if "clip_key" in manifest.columns:
        agg_items: dict = {"frames": ("clip_key", "count")}
        for col in ("video_id", "segment_id", "label", "split"):
            if col in manifest.columns:
                agg_items[col] = (col, "first")
        if "frame_number" in manifest.columns:
            agg_items["frame_min"] = ("frame_number", "min")
            agg_items["frame_max"] = ("frame_number", "max")
        clip_summary = (
            manifest.groupby("clip_key")
            .agg(**agg_items)
            .reset_index()
            .sort_values("frames", ascending=False)
        )

    return (
        {
            "videos": videos,
            "patients": patients,
            "segments": segments,
            "clips": clips,
            "frames_per_video_top15": {
                str(k): int(v) for k, v in manifest["video_id"].value_counts().head(15).items()
            }
            if "video_id" in manifest.columns
            else {},
            "frames_per_split": {
                str(k): int(v) for k, v in manifest["split"].value_counts().items()
            }
            if "split" in manifest.columns
            else {},
        },
        video_summary,
        clip_summary,
    )
