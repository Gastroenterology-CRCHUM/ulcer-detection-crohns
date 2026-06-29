"""
src/noninformative/predict.py
==============================
Inference utilities for the non-informative frame classifier.

Functions
---------
    predict_dataframe(df, model, extractor, ...)  → df with pred columns
    predict_video(video_path, model, extractor, ...) → frame-level results
    sample_level_aggregation(df)  → sample-level predictions
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.noninformative.features import (
    BottleneckExtractor,
    extract_all,
)
from src.noninformative.model import NonInformativeClassifier

LABEL_NAMES = {1: "Informative", 0: "Non-Informative"}


# ---------------------------------------------------------------------------
# Predict from manifest DataFrame
# ---------------------------------------------------------------------------


def predict_dataframe(
    df: pd.DataFrame,
    model: NonInformativeClassifier,
    use_bottleneck: bool = True,
    extractor: BottleneckExtractor | None = None,
    batch_size: int = 64,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run inference on all frames listed in a manifest DataFrame.

    The DataFrame must have an 'image_path' column.
    Adds columns: pred_label, pred_prob, pred_name, correct (if 'label' present).

    Args:
        df             : Manifest with 'image_path' (and optionally 'label').
        model          : Fitted NonInformativeClassifier.
        use_bottleneck : Include Inception-v3 bottleneck features.
        extractor      : Reuse an existing BottleneckExtractor.
        batch_size     : Batch size for bottleneck extraction.
        verbose        : Show progress bars.

    Returns:
        Copy of df with prediction columns added.
    """
    paths = df["image_path"].tolist()
    images = _load_images(paths, verbose=verbose)
    X = extract_all(
        images, use_bottleneck=use_bottleneck, bottleneck_extractor=extractor, verbose=verbose
    )

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= model.threshold).astype(int)

    result = df.copy()
    result["pred_label"] = preds
    result["pred_prob"] = probs.round(4)
    result["pred_name"] = [LABEL_NAMES[p] for p in preds]

    if "label" in df.columns:
        result["correct"] = (result["pred_label"] == result["label"]).astype(int)

    return result


# ---------------------------------------------------------------------------
# Predict from a video file
# ---------------------------------------------------------------------------


def predict_video(
    video_path: str | Path,
    model: NonInformativeClassifier,
    use_bottleneck: bool = True,
    extractor: BottleneckExtractor | None = None,
    every_n: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run inference on every N-th frame of a video.

    Returns a DataFrame with columns:
        frame_index, timestamp_s, pred_label, pred_prob, pred_name.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames, frame_indices = [], []
    idx = 0
    pbar = tqdm(total=total_frames // every_n, desc="Reading frames") if verbose else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_indices.append(idx)
            if pbar:
                pbar.update(1)
        idx += 1

    cap.release()
    if pbar:
        pbar.close()

    if not frames:
        return pd.DataFrame()

    X = extract_all(
        frames, use_bottleneck=use_bottleneck, bottleneck_extractor=extractor, verbose=verbose
    )
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= model.threshold).astype(int)

    return pd.DataFrame(
        {
            "frame_index": frame_indices,
            "timestamp_s": [i / fps for i in frame_indices],
            "pred_label": preds,
            "pred_prob": probs.round(4),
            "pred_name": [LABEL_NAMES[p] for p in preds],
        }
    )


# ---------------------------------------------------------------------------
# Sample-level aggregation
# ---------------------------------------------------------------------------


def sample_level_aggregation(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Aggregate frame-level predictions to sample level (mean_prob).

    Input df must have: sample_id, pred_prob, pred_label.
    Optionally: label (ground truth), cause.

    Returns one row per sample.
    """
    agg = {}

    for sample_id, grp in df.groupby("sample_id"):
        mean_prob = grp["pred_prob"].mean()
        sample_pred = int(mean_prob >= threshold)
        row = {
            "sample_id": sample_id,
            "n_frames": len(grp),
            "mean_prob": round(mean_prob, 4),
            "pred_label": sample_pred,
            "pred_name": LABEL_NAMES[sample_pred],
            "noninf_frame_pct": round((grp["pred_label"] == 0).mean(), 4),
        }

        if "label" in grp.columns:
            gt = int(grp["label"].mode().iloc[0])
            row["true_label"] = gt
            row["true_name"] = LABEL_NAMES[gt]
            row["correct"] = int(sample_pred == gt)

        if "cause" in grp.columns:
            cause = grp["cause"].dropna()
            row["cause"] = cause.mode().iloc[0] if not cause.empty else ""

        agg[sample_id] = row

    return pd.DataFrame(list(agg.values()))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_images(
    paths: list[str],
    verbose: bool = True,
) -> list[np.ndarray]:
    """Load a list of image paths as RGB numpy arrays."""
    images = []
    iterator = tqdm(paths, desc="Loading images") if verbose else paths
    for p in iterator:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {p}")
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images
