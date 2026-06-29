"""
scripts/noninformative/filter_raw_ulcer.py
==========================================
Run preprocessed Ulcer frames through the Non-Informative filter and
organize outputs into three categories.

Input frames must already be ROI-cropped (from data/ulcer/processed/
after the preprocess_frames step).

Expected input structure
------------------------
    processed_dir/
    └── Ulcer/
        └── vid_XX_XXXX/
            └── ulcer_X/
                └── *.png  (or .jpg)

Output structure
----------------
    output_dir/
    ├── informative/
    │   └── Ulcer/vid_XX_XXXX/ulcer_X/*.png   <- kept frames
    ├── non_informative/
    │   └── Ulcer/vid_XX_XXXX/ulcer_X/*.png   <- rejected frames
    ├── uncertain/
    │   └── Ulcer/vid_XX_XXXX/ulcer_X/*.png   <- review queue
    ├── predictions.csv                        <- all predictions
    └── stats.json                             <- per-video/segment statistics

Usage
-----
    python -m scripts.noninformative.filter_raw_ulcer \\
        --processed-dir data/ulcer/processed --model output/informative/models/rf_pipeline.pkl \\
        --output-dir output/ulcer/filtered

    # Without Inception bottleneck (faster)
    python -m scripts.noninformative.filter_raw_ulcer \\
        --processed-dir data/ulcer/processed --model output/informative/models/rf_pipeline.pkl \\
        --output-dir output/ulcer/filtered --no-bottleneck

    # Custom uncertainty threshold (default: 0.15 -> |prob - 0.5| < 0.15)
    python -m scripts.noninformative.filter_raw_ulcer \\
        --processed-dir data/ulcer/processed --model output/informative/models/rf_pipeline.pkl \\
        --output-dir output/ulcer/filtered --epsilon 0.10
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.paths import get_default_paths
from src.noninformative.features import (
    BottleneckExtractor,
    extract_all,
)
from src.noninformative.model import NonInformativeClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_EPSILON = 0.15  # |prob - 0.5| < epsilon -> uncertain


# ---------------------------------------------------------------------------
# Frame scanning
# ---------------------------------------------------------------------------


def scan_frames(processed_dir: Path) -> list[dict]:
    """Scan processed_dir/Ulcer/vid*/ulcer_*/*.png and return a list of dicts:
    image_path, vid_id, segment_id, filename
    """
    if not processed_dir.exists():
        raise FileNotFoundError(f"Directory not found: {processed_dir}")

    records = []
    for type_dir in sorted(processed_dir.iterdir()):
        if not type_dir.is_dir():
            continue
        for vid_dir in sorted(type_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            for seg_dir in sorted(vid_dir.iterdir()):
                if not seg_dir.is_dir():
                    continue
                for img_path in sorted(seg_dir.iterdir()):
                    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    records.append(
                        {
                            "image_path": img_path,
                            "vid_id": vid_dir.name,
                            "segment_id": seg_dir.name,
                            "filename": img_path.name,
                            "rel_path": img_path.relative_to(processed_dir),
                        }
                    )

    return records


# ---------------------------------------------------------------------------
# Batch feature extraction with ROI preprocessing
# ---------------------------------------------------------------------------


def extract_features(
    records: list[dict],
    use_bottleneck: bool,
    groups: list[str] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load images (already preprocessed) and extract features.

    Returns an array (N, D) and the list of loaded images.
    """
    print(f"\nLoading {len(records)} frames...")
    images = []
    for rec in tqdm(records, desc="Loading", unit="img"):
        bgr = cv2.imread(str(rec["image_path"]))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {rec['image_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(rgb)

    extractor = BottleneckExtractor() if use_bottleneck else None

    print(f"\nExtracting features (bottleneck={'yes' if use_bottleneck else 'no'})...")
    X = extract_all(
        images,
        use_bottleneck=use_bottleneck,
        bottleneck_extractor=extractor,
        verbose=True,
        groups=groups,
    )
    return X, images


# ---------------------------------------------------------------------------
# Classify and route frames
# ---------------------------------------------------------------------------


def classify_and_route(
    records: list[dict],
    images: list[np.ndarray],
    probs: np.ndarray,
    preds: np.ndarray,
    epsilon: float,
    output_dir: Path,
    dry_run: bool,
) -> pd.DataFrame:
    """Route each frame to one of three output categories:
      - informative     -> output_dir/informative/...
      - non_informative -> output_dir/non_informative/...
      - uncertain       -> output_dir/uncertain/...

    Returns the predictions DataFrame.
    """
    rows = []

    for rec, img_rgb, prob, pred in tqdm(
        zip(records, images, probs, preds),
        total=len(records),
        desc="Routing frames",
        unit="img",
    ):
        # Determine category
        uncertainty = abs(prob - 0.5)
        if uncertainty < epsilon:
            category = "uncertain"
        elif pred == 1:
            category = "informative"
        else:
            category = "non_informative"

        # Destination path
        dest = output_dir / category / rec["rel_path"]

        rows.append(
            {
                "image_path": str(rec["image_path"]),
                "vid_id": rec["vid_id"],
                "segment_id": rec["segment_id"],
                "filename": rec["filename"],
                "pred_label": int(pred),
                "pred_prob": round(float(prob), 4),
                "uncertainty": round(float(uncertainty), 4),
                "category": category,
                "dest_path": str(dest),
            }
        )

        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dest), img_bgr)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute global, per-video and per-segment statistics."""
    _total = len(df)

    def _counts(sub: pd.DataFrame) -> dict:
        return {
            "total": len(sub),
            "informative": int((sub["category"] == "informative").sum()),
            "non_informative": int((sub["category"] == "non_informative").sum()),
            "uncertain": int((sub["category"] == "uncertain").sum()),
            "inf_ratio": round((sub["category"] == "informative").mean(), 4),
        }

    stats = {
        "global": _counts(df),
        "by_video": {},
    }

    for vid_id, vid_df in df.groupby("vid_id"):
        stats["by_video"][vid_id] = {
            **_counts(vid_df),
            "segments": {},
        }
        for seg_id, seg_df in vid_df.groupby("segment_id"):
            stats["by_video"][vid_id]["segments"][seg_id] = _counts(seg_df)

    return stats


def print_stats(stats: dict) -> None:
    g = stats["global"]
    print("\n" + "=" * 60)
    print("GLOBAL STATISTICS")
    print("=" * 60)
    print(f"  Total frames       : {g['total']}")
    print(f"  Informative        : {g['informative']}  ({100 * g['inf_ratio']:.1f}%)")
    print(
        f"  Non-Informative    : {g['non_informative']}  "
        f"({100 * g['non_informative'] / g['total']:.1f}%)"
    )
    print(f"  Uncertain (review) : {g['uncertain']}  ({100 * g['uncertain'] / g['total']:.1f}%)")

    print("\nPER VIDEO")
    print("-" * 60)
    for vid_id, v in stats["by_video"].items():
        print(
            f"  {vid_id:30s}  "
            f"inf={v['informative']:4d}  "
            f"non_inf={v['non_informative']:4d}  "
            f"uncertain={v['uncertain']:3d}  "
            f"({100 * v['inf_ratio']:.0f}% inf)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)

    print("\n" + "=" * 60)
    print("Non-Informative Filter -- Preprocessed Ulcer Frames")
    print("=" * 60)
    print(f"  Source    : {processed_dir}")
    print(f"  RF model  : {model_path}")
    print(f"  Output    : {output_dir}")
    print(f"  Epsilon   : {args.epsilon}  (|prob-0.5| < epsilon -> uncertain)")
    if args.dry_run:
        print("  [DRY RUN] : no files will be copied")

    # 1. Scan
    print("\nScanning frames...")
    records = scan_frames(processed_dir)
    if not records:
        print("No frames found. Check --processed-dir.")
        return
    segments = {r["vid_id"] + "/" + r["segment_id"] for r in records}
    print(
        f"  {len(records)} frames found in "
        f"{len({r['vid_id'] for r in records})} video(s), "
        f"{len(segments)} segment(s)"
    )

    # 2. Load model
    model = NonInformativeClassifier.load(model_path)
    print(f"  RF threshold: {model.threshold:.3f}")

    # 3. Feature extraction
    # Compare feature names against the training cache
    cache_dir = get_default_paths().informative_models_dir
    cache_file = cache_dir / "features_cache.pkl"

    if not cache_file.exists():
        raise FileNotFoundError(f"Training cache not found: {cache_file}")
    else:
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        groups = cache.get("groups")
        use_bottleneck = cache.get("use_bottleneck", True)
        trained_names = cache.get("feature_names", [])
        print(f"Cached features  : {len(trained_names)}")
        print(f"Groups           : {groups if groups else 'all'}")
        print(f"Bottleneck       : {use_bottleneck}")
        print(f"Names (first 10) : {trained_names[:10]}")

    X, images = extract_features(
        records,
        use_bottleneck,
        groups=cache.get("groups", None) if cache else None,
    )
    print(f"Features extracted: {X.shape[1]}")
    print(f"Features expected : {model.scaler.n_features_in_}")

    # 4. Predict
    print("\nPredicting...")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= model.threshold).astype(int)
    print(f"  Informative    : {(preds == 1).sum()}")
    print(f"  Non-Informative: {(preds == 0).sum()}")
    uncertain_mask = np.abs(probs - 0.5) < args.epsilon
    print(f"  Uncertain      : {uncertain_mask.sum()}  (epsilon={args.epsilon})")

    # 5. Route frames
    print("\nCopying frames...")
    df = classify_and_route(
        records,
        images,
        probs,
        preds,
        epsilon=args.epsilon,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )

    # 6. Stats
    stats = compute_stats(df)
    print_stats(stats)

    # 7. Save outputs
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  CSV saved -> {csv_path}")

        stats_path = output_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved -> {stats_path}")

        # Review queue: separate CSV for uncertain frames
        uncertain_df = df[df["category"] == "uncertain"]
        if not uncertain_df.empty:
            review_path = output_dir / "review_queue.csv"
            uncertain_df.to_csv(review_path, index=False)
            print(f"  Review queue -> {review_path}  ({len(uncertain_df)} frames)")
    else:
        print("\n[DRY RUN] No files written.")
        print("Re-run without --dry-run to perform the copies.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _paths = get_default_paths()
    parser = argparse.ArgumentParser(
        description="Filter raw Ulcer frames with the Non-Informative classifier (Pipeline A)"
    )
    parser.add_argument(
        "--processed-dir",
        default=str(_paths.ulcer_processed_dir),
        help=f"Root of the preprocessed directory (default: {_paths.ulcer_processed_dir})",
    )
    parser.add_argument(
        "--model",
        default=str(_paths.informative_models_dir / "rf_pipeline.pkl"),
        help=f"Path to rf_pipeline.pkl (default: {_paths.informative_models_dir / 'rf_pipeline.pkl'})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_paths.filtered_dir),
        help=f"Output directory (default: {_paths.filtered_dir})",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Uncertainty threshold |prob-0.5| < epsilon (default: {DEFAULT_EPSILON})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without copying or writing any files",
    )
    args = parser.parse_args()
    main(args)
