"""Stage 5 — EDA for the ulcer detection dataset.

Combines frame-flow analysis (raw → processed → filtrated) with a full
dataset EDA covering class distribution, clip analysis, split quality,
ulcer-size balance, and optionally per-image statistics.

Input : data/ulcer/raw/, data/ulcer/processed/, data/ulcer/filtrated/
        data/ulcer/splits/dataset_manifest.csv
Output: results/ulcer/eda/
        ├── frame_flow.png
        ├── filter_outcomes.png
        ├── video_retention_ratio.png
        ├── class_distribution.png
        ├── clip_distribution.png
        ├── split_patients.png
        ├── split_clips.png
        ├── split_frames.png
        ├── split_class_balance.png
        ├── ulcer_size_distribution.png
        ├── top_videos_frame_count.png
        ├── sample_images.png
        ├── image_statistics.png         (optional --image-stats)
        ├── video_summary.csv
        ├── clip_summary.csv
        ├── split_label_table.csv
        ├── eda_report.txt
        └── statistics.json

Usage
-----
    python scripts/ulcer/eda.py
    python scripts/ulcer/eda.py --image-stats --image-sample-size 500
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from src.config.paths import get_default_paths
from src.data.constants import SIZE_LABELS, SIZE_ORDER
from src.data.eda_utils import (
    count_images,
    entity_summaries,
    manifest_quality_checks,
    plot_filter_outcomes,
    plot_frame_flow,
    plot_frames_per_video,
    plot_video_retention,
    split_diagnostics,
    to_serializable,
)
from src.data.pipeline_report import (
    annotation_duration_ulcer,
    collect_stage_stats,
    format_pipeline_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# DatasetEDA — ulcer-specific comprehensive analysis
# ---------------------------------------------------------------------------


class DatasetEDA:
    """Comprehensive EDA for the ulcer detection dataset.

    Terminology
    -----------
    patient  : one video_id (one colonoscopy recording)
    clip     : one segment within a video (ulcer_1, normal_2, ...)
    frame    : one individual image
    """

    def __init__(
        self,
        splits_dir: str = "data/ulcer/splits",
        output_dir: str = "results/ulcer/eda",
        fps: float = 1.0,
    ):
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        self.df: pd.DataFrame | None = None
        self.patient_info: dict | None = None
        self.split_info: dict | None = None
        self.image_stats: pd.DataFrame | None = None
        self._clip_df: pd.DataFrame | None = None

    def load_data(self) -> None:
        manifest_path = self.splits_dir / "dataset_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run scripts/ulcer/create_manifest.py first."
            )
        self.df = pd.read_csv(manifest_path)

        for candidate in (self.splits_dir / "patient_info.json",):
            if candidate.exists():
                with open(candidate) as f:
                    self.patient_info = json.load(f)
                break

        split_info_path = self.splits_dir / "split_info.json"
        if split_info_path.exists():
            with open(split_info_path) as f:
                self.split_info = json.load(f)

        logger.info(
            "Loaded %d frames | %d clips | %d patients",
            len(self.df),
            self.df["clip_key"].nunique() if "clip_key" in self.df.columns else 0,
            self.df["patient_id"].nunique() if "patient_id" in self.df.columns else 0,
        )

    # ------------------------------------------------------------------
    # Core statistics
    # ------------------------------------------------------------------

    def compute_dataset_statistics(self) -> dict:
        df = self.df

        n_patients_total = df["patient_id"].nunique()
        ulcer_patients = set(df[df["label"] == 1]["patient_id"].unique())
        non_ulcer_patients = set(df[df["label"] == 0]["patient_id"].unique())

        agg_dict: dict = {
            "patient_id": ("patient_id", "first"),
            "label": ("label", "max"),
            "n_frames": ("label", "count"),
            "split": ("split", "first"),
        }
        if "ulcer_size" in df.columns:
            agg_dict["ulcer_size"] = ("ulcer_size", "first")

        clip_df = df.groupby("clip_key").agg(**agg_dict).reset_index()
        self._clip_df = clip_df

        n_clips_ulcer = int((clip_df["label"] == 1).sum())
        n_clips_non_ulcer = int((clip_df["label"] == 0).sum())

        stats: dict = {
            "patients": {
                "total": n_patients_total,
                "ulcer_positive": len(ulcer_patients),
                "ulcer_negative": len(non_ulcer_patients - ulcer_patients),
            },
            "clips": {
                "total": len(clip_df),
                "ulcer_positive": n_clips_ulcer,
                "ulcer_negative": n_clips_non_ulcer,
                "frames_per_clip": {
                    "mean": clip_df["n_frames"].mean(),
                    "std": clip_df["n_frames"].std(),
                    "min": int(clip_df["n_frames"].min()),
                    "max": int(clip_df["n_frames"].max()),
                    "median": clip_df["n_frames"].median(),
                    "fps": self.fps,
                    "duration_s_mean": clip_df["n_frames"].mean() / self.fps,
                    "duration_s_std": clip_df["n_frames"].std() / self.fps,
                    "duration_s_min": clip_df["n_frames"].min() / self.fps,
                    "duration_s_max": clip_df["n_frames"].max() / self.fps,
                    "duration_s_median": clip_df["n_frames"].median() / self.fps,
                },
            },
            "frames": {
                "total": len(df),
                "ulcer_positive": int((df["label"] == 1).sum()),
                "ulcer_negative": int((df["label"] == 0).sum()),
            },
        }

        if "split" in df.columns:
            split_stats = {}
            for split in SPLITS:
                s_df = df[df["split"] == split]
                s_clips = clip_df[clip_df["split"] == split]
                if s_df.empty:
                    continue
                entry: dict = {
                    "n_patients": s_df["patient_id"].nunique(),
                    "n_clips": len(s_clips),
                    "n_clips_ulcer": int((s_clips["label"] == 1).sum()),
                    "n_clips_non_ulcer": int((s_clips["label"] == 0).sum()),
                    "n_frames": len(s_df),
                    "n_frames_ulcer": int((s_df["label"] == 1).sum()),
                    "n_frames_non_ulcer": int((s_df["label"] == 0).sum()),
                    "pct_of_total": len(s_df) / len(df) * 100,
                }
                if "ulcer_size" in s_clips.columns:
                    ulcer_clips = s_clips[s_clips["label"] == 1]
                    size_series = (
                        ulcer_clips["ulcer_size"]
                        .map(
                            lambda x: (
                                SIZE_LABELS.get(int(x), "unknown") if pd.notna(x) else "unknown"
                            )
                        )
                        .value_counts()
                        .reindex(SIZE_ORDER, fill_value=0)
                    )
                    entry["ulcer_size_clips"] = size_series.to_dict()
                    n_u = entry["n_clips_ulcer"]
                    entry["ulcer_size_clips_pct"] = {
                        k: (v / n_u * 100 if n_u else 0)
                        for k, v in entry["ulcer_size_clips"].items()
                    }
                split_stats[split] = entry
            stats["per_split"] = split_stats

        if "ulcer_size" in df.columns:
            ulcer_clips = clip_df[clip_df["label"] == 1]
            size_counts = (
                ulcer_clips["ulcer_size"]
                .map(lambda x: SIZE_LABELS.get(int(x), "unknown") if pd.notna(x) else "unknown")
                .value_counts()
                .reindex(SIZE_ORDER, fill_value=0)
            )
            stats["ulcer_size"] = {k: int(v) for k, v in size_counts.items()}

        if self.split_info and "stratification" in self.split_info:
            stats["stratification_audit"] = self.split_info["stratification"]

        return stats

    def compute_image_statistics(self, sample_size: int | None = None, n_workers: int = 4) -> dict:
        sample_df = (
            self.df.sample(n=sample_size, random_state=42)
            if sample_size and len(self.df) > sample_size
            else self.df
        )

        def _analyze(row):
            try:
                p = Path(row["image_path"])
                size_kb = p.stat().st_size / 1024
                with Image.open(p) as img:
                    w, h = img.size
                    mode = img.mode
                return dict(
                    class_name=row["class_name"],
                    width=w,
                    height=h,
                    aspect_ratio=w / h if h else 0,
                    file_size_kb=size_kb,
                    mode=mode,
                )
            except Exception as exc:
                logger.warning("Error reading %s: %s", row["image_path"], exc)
                return None

        records = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_analyze, row): i for i, row in sample_df.iterrows()}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Image stats"):
                r = fut.result()
                if r:
                    records.append(r)

        if not records:
            return {}

        img_df = pd.DataFrame(records)
        self.image_stats = img_df

        return {
            "dimensions": {
                ax: {
                    "mean": img_df[ax].mean(),
                    "std": img_df[ax].std(),
                    "min": int(img_df[ax].min()),
                    "max": int(img_df[ax].max()),
                    "n_unique": img_df[ax].nunique(),
                }
                for ax in ("width", "height")
            },
            "aspect_ratio": {
                "mean": img_df["aspect_ratio"].mean(),
                "std": img_df["aspect_ratio"].std(),
            },
            "file_size_kb": {
                "mean": img_df["file_size_kb"].mean(),
                "std": img_df["file_size_kb"].std(),
                "min": img_df["file_size_kb"].min(),
                "max": img_df["file_size_kb"].max(),
            },
            "color_modes": img_df["mode"].value_counts().to_dict(),
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_class_distribution(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        counts = self.df["class_name"].value_counts()
        bars = ax.bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="black")
        for bar, n in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"{n:,}\n({n / len(self.df) * 100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title("Frame-level class distribution")
        ax.set_ylabel("Frames")

        ax = axes[1]
        clip_counts = self._clip_df["label"].map({0: "NonUlcer", 1: "Ulcer"}).value_counts()
        bars = ax.bar(
            clip_counts.index, clip_counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="black"
        )
        for bar, n in zip(bars, clip_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{n}\n({n / len(self._clip_df) * 100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title("Clip-level class distribution")
        ax.set_ylabel("Clips")

        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_clip_distribution(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for label, color, name in [(1, "#e74c3c", "Ulcer"), (0, "#2ecc71", "NonUlcer")]:
            data = self._clip_df[self._clip_df["label"] == label]["n_frames"]
            ax.hist(data, bins=25, alpha=0.6, color=color, edgecolor="black", label=name)
        ax.axvline(
            self._clip_df["n_frames"].mean(),
            color="black",
            linestyle="--",
            label=f"Mean = {self._clip_df['n_frames'].mean():.0f}",
        )
        ax.set_xlabel("Frames per clip")
        ax.set_ylabel("Clips")
        ax.set_title("Frame count distribution per clip")
        ax.legend()

        ax = axes[1]
        if self.patient_info:
            types: dict[str, int] = defaultdict(int)
            for info in self.patient_info.values():
                if info["has_ulcer"] and info["has_non_ulcer"]:
                    types["Mixed"] += 1
                elif info["has_ulcer"]:
                    types["Ulcer only"] += 1
                else:
                    types["NonUlcer only"] += 1
            ax.pie(
                types.values(),
                labels=types.keys(),
                autopct="%1.1f%%",
                colors=["#3498db", "#e74c3c", "#2ecc71"],
                explode=[0.02] * len(types),
            )
            ax.set_title("Patient type distribution")
        else:
            ax.text(0.5, 0.5, "patient_info.json not found", ha="center", va="center")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "clip_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_split_analysis(self) -> None:
        if "split" not in self.df.columns:
            return

        present_splits = [s for s in SPLITS if s in self.df["split"].values]
        colors = {"train": "#3498db", "val": "#2ecc71", "test": "#e74c3c"}

        # --- patients per split ---
        fig, ax = plt.subplots(figsize=(6, 5))
        pt_counts = self.df.groupby("split")["patient_id"].nunique().reindex(present_splits)
        bars = ax.bar(
            pt_counts.index,
            pt_counts.values,
            color=[colors[s] for s in pt_counts.index],
            edgecolor="black",
        )
        for bar, n in zip(bars, pt_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(n),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_title("Patients per split")
        ax.set_ylabel("Patients")
        plt.tight_layout()
        plt.savefig(self.output_dir / "split_patients.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- clips per split ---
        fig, ax = plt.subplots(figsize=(6, 5))
        clip_split = (
            self._clip_df.groupby(["split", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(present_splits)
            .rename(columns={0: "NonUlcer", 1: "Ulcer"})
        )
        clip_split.plot(
            kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black"
        )
        ax.set_title("Clips per split")
        ax.set_ylabel("Clips")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Class")
        for container in ax.containers:
            ax.bar_label(
                container, label_type="center", fontsize=8, color="white", fontweight="bold"
            )
        plt.tight_layout()
        plt.savefig(self.output_dir / "split_clips.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- frames per split ---
        fig, ax = plt.subplots(figsize=(6, 5))
        fr_counts = self.df.groupby("split").size().reindex(present_splits)
        bars = ax.bar(
            fr_counts.index,
            fr_counts.values,
            color=[colors[s] for s in fr_counts.index],
            edgecolor="black",
        )
        for bar, n in zip(bars, fr_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{n:,}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title("Frames per split")
        ax.set_ylabel("Frames")
        plt.tight_layout()
        plt.savefig(self.output_dir / "split_frames.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- class balance per split ---
        fig, ax = plt.subplots(figsize=(6, 5))
        balance = (
            self.df.groupby(["split", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(present_splits)
            .rename(columns={0: "NonUlcer", 1: "Ulcer"})
        )
        balance_pct = balance.div(balance.sum(axis=1), axis=0) * 100
        balance_pct.plot(
            kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black"
        )
        ax.axhline(50, color="black", linestyle="--", alpha=0.4)
        ax.set_title("Class balance per split (%)")
        ax.set_ylabel("%")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Class")
        plt.tight_layout()
        plt.savefig(self.output_dir / "split_class_balance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_ulcer_size_distribution(self) -> None:
        if "ulcer_size" not in self.df.columns:
            return

        present_splits = [s for s in SPLITS if s in self.df["split"].values]
        ulcer_clips = self._clip_df[self._clip_df["label"] == 1].copy()
        ulcer_clips["size_label"] = ulcer_clips["ulcer_size"].map(
            lambda x: SIZE_LABELS.get(int(x), "unknown") if pd.notna(x) else "unknown"
        )

        size_colors = {
            "<5mm": "#f39c12",
            "5-20mm": "#e74c3c",
            ">20mm": "#8e44ad",
            "unknown": "#95a5a6",
        }
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        ax = axes[0]
        counts = ulcer_clips["size_label"].value_counts().reindex(SIZE_ORDER, fill_value=0)
        total = counts.sum()
        bars = ax.bar(
            counts.index,
            counts.values,
            color=[size_colors[s] for s in counts.index],
            edgecolor="black",
            alpha=0.85,
        )
        for bar, n in zip(bars, counts.values):
            pct = n / total * 100 if total else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{n}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title("Ulcer size — overall (clips)")
        ax.set_ylabel("Clips")

        ax = axes[1]
        size_per_split = (
            ulcer_clips.groupby(["split", "size_label"])
            .size()
            .unstack(fill_value=0)
            .reindex(present_splits)
            .reindex(SIZE_ORDER, axis=1, fill_value=0)
        )
        size_per_split.plot(
            kind="bar",
            ax=ax,
            color=[size_colors[s] for s in SIZE_ORDER],
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title("Ulcer size per split (absolute)")
        ax.set_ylabel("Clips")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Size", bbox_to_anchor=(1.01, 1), loc="upper left")

        ax = axes[2]
        size_pct = size_per_split.div(size_per_split.sum(axis=1).replace(0, np.nan), axis=0) * 100
        size_pct.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[size_colors[s] for s in SIZE_ORDER],
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title("Ulcer size per split (% of ulcer clips)")
        ax.set_ylabel("%")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Size", bbox_to_anchor=(1.01, 1), loc="upper left")

        plt.suptitle("Ulcer size distribution across splits", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "ulcer_size_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_image_statistics(self) -> None:
        if self.image_stats is None or self.image_stats.empty:
            return

        img_df = self.image_stats
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        color_map = img_df["class_name"].map({"Ulcer": "#e74c3c", "NonUlcer": "#2ecc71"})
        ax.scatter(
            img_df["width"],
            img_df["height"],
            c=color_map,
            alpha=0.4,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title("Image dimensions")
        ax.legend(
            handles=[
                plt.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="#2ecc71", label="NonUlcer"
                ),
                plt.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="#e74c3c", label="Ulcer"
                ),
            ]
        )

        ax = axes[1]
        for cls, color in [("NonUlcer", "#2ecc71"), ("Ulcer", "#e74c3c")]:
            ax.hist(
                img_df[img_df["class_name"] == cls]["file_size_kb"],
                bins=30,
                alpha=0.5,
                color=color,
                edgecolor="black",
                label=cls,
            )
        ax.set_xlabel("File size (KB)")
        ax.set_ylabel("Count")
        ax.set_title("File size distribution")
        ax.legend()

        ax = axes[2]
        ax.hist(img_df["aspect_ratio"], bins=30, edgecolor="black", alpha=0.7, color="#3498db")
        ax.axvline(1.0, color="red", linestyle="--", label="Square (1:1)")
        ax.set_xlabel("Aspect ratio (W/H)")
        ax.set_ylabel("Count")
        ax.set_title("Aspect ratio distribution")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "image_statistics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_sample_images(self, n_samples: int = 4) -> None:
        fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
        for row_i, class_name in enumerate(["NonUlcer", "Ulcer"]):
            class_df = self.df[self.df["class_name"] == class_name]
            samples = class_df.sample(n=min(n_samples, len(class_df)), random_state=42)
            for col_i, (_, rec) in enumerate(samples.iterrows()):
                ax = axes[row_i, col_i]
                try:
                    ax.imshow(Image.open(rec["image_path"]))
                    title = f"{class_name}\n{rec['patient_id']} / {rec['segment_id']}"
                    if (
                        class_name == "Ulcer"
                        and "ulcer_size" in rec
                        and pd.notna(rec["ulcer_size"])
                    ):
                        title += f"\n{SIZE_LABELS.get(int(rec['ulcer_size']), '?')}"
                    ax.set_title(title, fontsize=8)
                except Exception as exc:
                    ax.text(0.5, 0.5, str(exc), ha="center", va="center", fontsize=7)
                ax.axis("off")
        plt.suptitle("Sample images", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_images.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, stats: dict, img_stats: dict) -> str:
        lines: list[str] = []
        W = 76

        def h1(title: str) -> None:
            lines.append("=" * W)
            lines.append(title.center(W))
            lines.append("=" * W)

        def h2(title: str) -> None:
            lines.append(f"\n{title}")
            lines.append("-" * len(title))

        h1("ULCER DETECTION - EXPLORATORY DATA ANALYSIS REPORT")

        h2("1. DATASET OVERVIEW")
        p, c, f = stats["patients"], stats["clips"], stats["frames"]
        lines.append(f"  {'':40} {'Total':>8} {'Ulcer+':>8} {'Ulcer-':>8}")
        lines.append("  " + "-" * 64)
        lines.append(
            f"  {'Patients':<40} {p['total']:>8} {p['ulcer_positive']:>8} {p['ulcer_negative']:>8}"
        )
        lines.append(
            f"  {'Clips':<40} {c['total']:>8} {c['ulcer_positive']:>8} {c['ulcer_negative']:>8}"
        )
        lines.append(
            f"  {'Frames':<40} {f['total']:>8,} {f['ulcer_positive']:>8,} {f['ulcer_negative']:>8,}"
        )
        fpc = c["frames_per_clip"]
        lines.append(
            f"\n  Frames/clip  mean {fpc['mean']:.1f} +/- {fpc['std']:.1f}"
            f"   [min {fpc['min']} - max {fpc['max']} | median {fpc['median']:.0f}]"
        )
        lines.append(
            f"  Duration (s) mean {fpc['duration_s_mean']:.1f} +/- {fpc['duration_s_std']:.1f}"
            f"   [min {fpc['duration_s_min']:.1f} - max {fpc['duration_s_max']:.1f}"
            f" | median {fpc['duration_s_median']:.1f}]"
            f"   (@ {fpc['fps']} FPS)"
        )

        if "ulcer_size" in stats:
            h2("2. ULCER SIZE -- OVERALL (ulcer clips)")
            n_u = c["ulcer_positive"]
            lines.append(f"  {'Category':<20} {'Clips':>8} {'%':>8}")
            lines.append("  " + "-" * 38)
            for cat in SIZE_ORDER:
                n = stats["ulcer_size"].get(cat, 0)
                pct = n / n_u * 100 if n_u else 0
                lines.append(f"  {cat:<20} {n:>8} {pct:>7.1f}%")

        if "per_split" in stats:
            h2("3. SPLIT STATISTICS")
            present = {k: v for k, v in stats["per_split"].items() if v}
            col_w = 12
            header = f"  {'':32}" + "".join(f"{s:>{col_w}}" for s in SPLITS)
            lines.append(header)
            lines.append("  " + "-" * (32 + col_w * len(SPLITS)))
            for metric, label in [
                ("n_patients", "Patients"),
                ("n_clips", "Clips (total)"),
                ("n_clips_ulcer", "  Ulcer clips"),
                ("n_clips_non_ulcer", "  NonUlcer clips"),
                ("n_frames", "Frames (total)"),
                ("n_frames_ulcer", "  Ulcer frames"),
                ("n_frames_non_ulcer", "  NonUlcer frames"),
            ]:
                vals = [present.get(s, {}).get(metric, "-") for s in SPLITS]
                lines.append(
                    f"  {label:<32}"
                    + "".join(
                        f"{v:>{col_w},}" if isinstance(v, int) else f"{v!s:>{col_w}}" for v in vals
                    )
                )
            pcts = [f"{present.get(s, {}).get('pct_of_total', 0):.1f}%" for s in SPLITS]
            lines.append(f"  {'% of total frames':<32}" + "".join(f"{p:>{col_w}}" for p in pcts))
            lines.append("")
            lines.append(
                f"  {'Class split distribution':<32}" + "".join(f"{s:>{col_w}}" for s in SPLITS)
            )
            lines.append("  " + "-" * (32 + col_w * len(SPLITS)))
            for class_key, label in [
                ("n_clips_ulcer", "  Ulcer clips"),
                ("n_clips_non_ulcer", "  NonUlcer clips"),
                ("n_frames_ulcer", "  Ulcer frames"),
                ("n_frames_non_ulcer", "  NonUlcer frames"),
            ]:
                class_total = sum(present.get(s, {}).get(class_key, 0) for s in SPLITS)
                vals = []
                for s in SPLITS:
                    n = present.get(s, {}).get(class_key, 0)
                    vals.append(f"{n / class_total * 100:.1f}%" if class_total else "-")
                lines.append(f"  {label:<32}" + "".join(f"{v:>{col_w}}" for v in vals))

        if "per_split" in stats and any(
            "ulcer_size_clips" in v for v in stats["per_split"].values()
        ):
            h2("4. ULCER SIZE BALANCE ACROSS SPLITS (ulcer clips)")
            lines.append(f"  {'':20}" + "".join(f"{'  ' + s + ' (n / %)':>22}" for s in SPLITS))
            lines.append("  " + "-" * (20 + 22 * len(SPLITS)))
            for cat in SIZE_ORDER:
                row_parts = []
                for split in SPLITS:
                    sp = stats["per_split"].get(split, {})
                    n = sp.get("ulcer_size_clips", {}).get(cat, 0)
                    pct = sp.get("ulcer_size_clips_pct", {}).get(cat, 0.0)
                    row_parts.append(f"{n:>5} ({pct:>4.1f}%)")
                lines.append(f"  {cat:<20}" + "     ".join(row_parts))
            lines.append("")
            lines.append("  Note: val merges with train during cross-validation.")

        h2("5. DATA LEAKAGE CHECK (patient-level)")
        if "split" in self.df.columns:
            split_patients = {s: set(self.df[self.df["split"] == s]["patient_id"]) for s in SPLITS}
            ok = True
            for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
                overlap = split_patients.get(s1, set()) & split_patients.get(s2, set())
                if overlap:
                    lines.append(f"  [!] {s1}/{s2} overlap: {len(overlap)} patient(s): {overlap}")
                    ok = False
            if ok:
                lines.append("  [OK] No patient overlap between any splits.")
        else:
            lines.append("  No split column found.")

        if "stratification_audit" in stats:
            h2("6. STRATIFICATION AUDIT")
            audit = stats["stratification_audit"]
            lines.append(f"  Method : {audit.get('method', 'N/A')}")
            rare = audit.get("rare_strata_patients", [])
            if rare:
                lines.append(f"  [!] {len(rare)} patient(s) in rare strata -- assigned manually:")
                lines.append(f"      {rare}")
                lines.append("  Priority order applied: train > test > val")
            else:
                lines.append("  [OK] No rare strata — all patients split via stratified sampling.")
            lines.append("\n  Strata (ulcer_presence x dominant_ulcer_size):")
            for stratum, count in sorted(audit.get("strata_counts", {}).items()):
                flag = "  [rare]" if count < 3 else ""
                lines.append(f"    {stratum:<38} {count:>3} patient(s){flag}")

        if img_stats:
            h2("7. IMAGE STATISTICS")
            dims = img_stats.get("dimensions", {})
            for ax_name in ("width", "height"):
                d = dims.get(ax_name, {})
                lines.append(
                    f"  {ax_name.capitalize():<10}"
                    f"mean {d.get('mean', 0):.0f} +/- {d.get('std', 0):.0f} px   "
                    f"[{d.get('min', 0)} - {d.get('max', 0)}]   "
                    f"{d.get('n_unique', 0)} unique"
                )
            fs = img_stats.get("file_size_kb", {})
            lines.append(
                f"  File size   mean {fs.get('mean', 0):.1f} +/- {fs.get('std', 0):.1f} KB   "
                f"[{fs.get('min', 0):.1f} - {fs.get('max', 0):.1f}]"
            )

        lines.append("\n" + "=" * W)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        compute_image_stats: bool = False,
        image_sample_size: int | None = None,
    ) -> tuple[dict, dict]:
        logger.info("Starting ulcer EDA...")
        self.load_data()

        stats = self.compute_dataset_statistics()
        img_stats = (
            self.compute_image_statistics(sample_size=image_sample_size)
            if compute_image_stats
            else {}
        )

        self.plot_class_distribution()
        self.plot_clip_distribution()
        self.plot_split_analysis()
        self.plot_ulcer_size_distribution()
        if img_stats:
            self.plot_image_statistics()
        self.plot_sample_images()

        plot_frames_per_video(self.df, self.output_dir)

        quality = manifest_quality_checks(self.df)
        split_diag = split_diagnostics(self.df)
        entities, video_summary_df, clip_summary_df = entity_summaries(self.df)

        if {"split", "label"}.issubset(set(self.df.columns)):
            split_label_table = pd.crosstab(self.df["split"], self.df["label"], dropna=False)
            split_label_table.to_csv(self.output_dir / "split_label_table.csv")
        if not video_summary_df.empty:
            video_summary_df.to_csv(self.output_dir / "video_summary.csv", index=False)
        if not clip_summary_df.empty:
            clip_summary_df.to_csv(self.output_dir / "clip_summary.csv", index=False)

        report = self.generate_report(stats, img_stats)
        print(report)
        (self.output_dir / "eda_report.txt").write_text(report, encoding="utf-8")

        all_stats = to_serializable(
            {
                "dataset_stats": stats,
                "image_stats": img_stats,
                "manifest_quality": quality,
                "split_diagnostics": split_diag,
                "entities": entities,
            }
        )
        with open(self.output_dir / "statistics.json", "w", encoding="utf-8") as fh:
            json.dump(all_stats, fh, indent=2)

        logger.info("EDA complete — outputs saved to %s", self.output_dir)
        return stats, img_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    paths = get_default_paths()
    parser = argparse.ArgumentParser(description="Run EDA on the ulcer detection dataset.")
    parser.add_argument("--raw-dir", type=str, default=str(paths.ulcer_raw_dir))
    parser.add_argument("--processed-dir", type=str, default=str(paths.ulcer_processed_dir))
    parser.add_argument("--filtrated-dir", type=str, default=str(paths.ulcer_filtrated_dir))
    parser.add_argument("--splits-dir", type=str, default=str(paths.ulcer_splits_dir))
    parser.add_argument("--output-dir", type=str, default=str(paths.results_eda_dir))
    parser.add_argument(
        "--excel",
        type=str,
        default=str(paths.ulcer_raw_dir / "Ulcer and Non-Ulcer Timestamps.xlsx"),
        help="Path to the ulcer annotation Excel workbook (for annotation duration stats).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Extraction frame rate used to convert frame counts to clip duration (default 10.0).",
    )
    parser.add_argument(
        "--image-stats",
        action="store_true",
        help="Compute per-image dimension/file-size statistics (slow).",
    )
    parser.add_argument(
        "--image-sample-size",
        type=int,
        default=None,
        help="Subsample N images for image-stats (default: all).",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    filtrated_dir = Path(args.filtrated_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_count = count_images(raw_dir)
    processed_count = count_images(processed_dir)
    filtrated_count = count_images(filtrated_dir)

    plot_frame_flow(
        raw_count, processed_count, filtrated_count, output_dir, title="Ulcer Frame Flow"
    )

    pred_df = pd.DataFrame()
    pred_path = filtrated_dir / "predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        if "video_id" not in pred_df.columns and "relative_path" in pred_df.columns:
            pred_df["video_id"] = pred_df["relative_path"].str.split("/").str[1]

    plot_filter_outcomes(pred_df, output_dir)
    plot_video_retention(pred_df, output_dir)

    # ── Preprocessing pipeline report ────────────────────────────────────
    ann_stats = annotation_duration_ulcer(Path(args.excel))
    stage_stats = [
        ("Raw", collect_stage_stats(raw_dir, args.fps)),
        ("Processed", collect_stage_stats(processed_dir, args.fps)),
        ("Filtrated", collect_stage_stats(filtrated_dir, args.fps)),
    ]
    prep_report = format_pipeline_report(
        "ULCER PREPROCESSING PIPELINE REPORT",
        ann_stats,
        stage_stats,
        fps=args.fps,
        label_order=["Ulcer", "NonUlcer"],
    )
    (output_dir / "preprocessing_report.txt").write_text(prep_report, encoding="utf-8")
    print(prep_report)

    eda = DatasetEDA(splits_dir=args.splits_dir, output_dir=args.output_dir, fps=args.fps)
    eda.run_full_analysis(
        compute_image_stats=args.image_stats,
        image_sample_size=args.image_sample_size,
    )

    removed = max(processed_count - filtrated_count, 0)
    kept_ratio = filtrated_count / processed_count if processed_count else 0.0
    print("=" * 72)
    print("ULCER EDA DONE")
    print("=" * 72)
    print(f"Raw frames       : {raw_count:,}")
    print(f"Processed frames : {processed_count:,}")
    print(f"Filtrated frames : {filtrated_count:,}  (removed: {removed:,}, kept: {kept_ratio:.1%})")
    print(f"Output dir       : {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main(build_parser().parse_args())
