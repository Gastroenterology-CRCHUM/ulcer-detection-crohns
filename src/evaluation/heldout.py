"""
src/evaluation/heldout.py
-------------------------
The temporal held-out cohort and the CV models' predictions on it.

Public API
----------
load_cohort(manifest_path, frame_mask=None) -> HeldoutCohort
load_informative_mask(manifest_path, mask_path) -> bool array, manifest row order
aggregate_to_clip(frame_probs, frame_clip_keys, clip_order) -> clip probabilities
point_metrics(labels, probs, threshold) -> {auroc, accuracy, f1, precision, sensitivity, specificity}
load_predictions(client, runs, full, cohort, frame_mask) -> ConfigPredictions | None
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.mlflow_io import CVRuns, artifact_paths, download_npy

METRICS = ("auroc", "f1", "precision", "sensitivity", "specificity", "accuracy")


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldoutCohort:
    """Frame rows in manifest order, clips in sorted `clip_order`.

    A clip's label is the majority (mode) of its frames' labels.
    """

    clip_order: list[str]
    clip_labels: np.ndarray
    clip_patient_ids: np.ndarray
    frame_clip_keys: np.ndarray
    frame_labels: np.ndarray
    frame_patient_ids: np.ndarray

    def clip_labels_in(self, order) -> np.ndarray:
        """Clip labels re-ordered to `order` (e.g. first-appearance order)."""
        lookup = dict(zip(self.clip_order, self.clip_labels))
        return np.array([lookup[c] for c in order], dtype=int)


def load_cohort(manifest_path, frame_mask: np.ndarray | None = None) -> HeldoutCohort:
    """`frame_mask` (manifest row order, True = keep) drops frames first, so a clip with
    no surviving frame disappears and clip labels are voted on surviving frames only."""
    df = pd.read_csv(manifest_path)
    if frame_mask is not None:
        df = df[frame_mask].reset_index(drop=True)
    clip_order = sorted(df["clip_key"].unique().tolist())
    label_map = df.groupby("clip_key")["label"].apply(lambda s: int(s.mode()[0]))
    patient_map = df.groupby("clip_key")["patient_id"].first()
    return HeldoutCohort(
        clip_order=clip_order,
        clip_labels=np.array([label_map[c] for c in clip_order], dtype=int),
        clip_patient_ids=np.array([patient_map[c] for c in clip_order]),
        frame_clip_keys=df["clip_key"].to_numpy(),
        frame_labels=df["label"].to_numpy(),
        frame_patient_ids=df["patient_id"].to_numpy(),
    )


def load_informative_mask(manifest_path, mask_path) -> np.ndarray:
    """Boolean keep-mask in manifest row order, from a `relative_path,informative` CSV
    (the informative-frame classifier of scripts/noninformative/filter_frames.py).
    Frames missing from the CSV are kept, with a warning."""
    manifest = pd.read_csv(manifest_path)
    mask = pd.read_csv(mask_path)
    keep = manifest["relative_path"].map(dict(zip(mask["relative_path"], mask["informative"])))
    n_unknown = int(keep.isna().sum())
    if n_unknown:
        print(
            f"  [warn] {n_unknown} manifest frame(s) not found in the informative mask, "
            "keeping them by default."
        )
    return keep.fillna(1).astype(bool).to_numpy()


# ---------------------------------------------------------------------------
# Aggregation and metrics
# ---------------------------------------------------------------------------


def aggregate_to_clip(
    frame_probs: np.ndarray, frame_clip_keys: np.ndarray, clip_order
) -> np.ndarray:
    """Mean frame probability per clip, returned in `clip_order`."""
    df = pd.DataFrame({"clip_key": frame_clip_keys, "prob": frame_probs})
    return df.groupby("clip_key", sort=False)["prob"].mean().reindex(clip_order).to_numpy()


def point_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Metrics at a fixed threshold (AUROC is threshold-free). Frame or clip level."""
    preds = (probs >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "sensitivity": recall_score(labels, preds, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


@dataclass
class ConfigPredictions:
    """One configuration's held-out predictions, after the frame filter."""

    ensemble_frame_probs: np.ndarray
    ensemble_clip_probs: np.ndarray
    best_fold_clip_probs: np.ndarray
    fold_clip_probs: list[np.ndarray]
    fold_metrics: list[dict]  # clip level, fold clip threshold
    fold_metrics_frame: list[dict]  # frame level, fold frame threshold
    fold_thresholds: list[float]  # clip thresholds
    cv_mean_threshold: float  # frame level, applied to the ensemble
    cv_mean_clip_threshold: float  # clip level, applied to the ensemble


def load_predictions(
    client: MlflowClient,
    runs: CVRuns,
    full: HeldoutCohort,
    cohort: HeldoutCohort,
    frame_mask: np.ndarray | None = None,
) -> ConfigPredictions | None:
    """Fold and ensemble held-out predictions of one configuration.

    Each fold's `predictions/heldout/*probs*.npy` is indexed like the FULL manifest;
    the sibling labels artifact must equal the manifest labels row for row, otherwise
    clip aggregation would be misaligned and the configuration is skipped (None).
    `cohort` is `full` after `frame_mask`; the mask is applied to the probabilities
    only after every artifact has been validated against `full`.
    """
    fold_probs, clip_thresholds, frame_thresholds = [], [], []
    for child in runs.children:
        name = child.data.tags.get("mlflow.runName", child.info.run_id[:8])
        paths = artifact_paths(client, child.info.run_id, "predictions/heldout")
        probs_paths = [p for p in paths if "probs" in p]
        labels_paths = [p for p in paths if "labels" in p]
        if not probs_paths or not labels_paths:
            print(f"    [warn] {name}: missing heldout probs/labels artifact, skipping config.")
            return None
        probs = download_npy(client, child.info.run_id, probs_paths[0])
        labels = download_npy(client, child.info.run_id, labels_paths[0])
        n = len(full.frame_clip_keys)
        if len(probs) != n or len(labels) != n:
            print(
                f"    [warn] {name}: {len(probs)} probs / {len(labels)} labels, expected "
                f"{n}, skipping config."
            )
            return None
        if not np.array_equal(labels, full.frame_labels):
            print(
                f"    [warn] {name}: heldout labels do not match the manifest row order "
                f"(frame/clip alignment would be wrong), skipping config."
            )
            return None
        fold_probs.append(probs)
        clip_thresholds.append(float(child.data.metrics.get("fold_optimal_clip_threshold", 0.5)))
        frame_thresholds.append(float(child.data.metrics.get("fold_optimal_threshold", 0.5)))

    parent_id = runs.parent.info.run_id
    try:
        best_probs = download_npy(client, parent_id, "predictions/heldout_best_fold_probs.npy")
        best_labels = download_npy(client, parent_id, "predictions/heldout_best_fold_labels.npy")
        if len(best_probs) != len(full.frame_clip_keys) or not np.array_equal(
            best_labels, full.frame_labels
        ):
            raise ValueError("best-fold heldout artifact shape/order mismatch")
    except Exception as exc:
        print(
            f"    [warn] best-fold heldout probs unavailable ({exc}), using ensemble as fallback."
        )
        best_probs = None

    keep = np.ones(len(full.frame_clip_keys), dtype=bool) if frame_mask is None else frame_mask
    fold_probs = [p[keep] for p in fold_probs]

    fold_clip_probs = [
        aggregate_to_clip(p, cohort.frame_clip_keys, cohort.clip_order) for p in fold_probs
    ]
    ensemble_frame_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
    ensemble_clip_probs = aggregate_to_clip(
        ensemble_frame_probs, cohort.frame_clip_keys, cohort.clip_order
    )
    best_fold_clip_probs = (
        ensemble_clip_probs
        if best_probs is None
        else aggregate_to_clip(best_probs[keep], cohort.frame_clip_keys, cohort.clip_order)
    )

    parent_metrics = runs.parent.data.metrics
    return ConfigPredictions(
        ensemble_frame_probs=ensemble_frame_probs,
        ensemble_clip_probs=ensemble_clip_probs,
        best_fold_clip_probs=best_fold_clip_probs,
        fold_clip_probs=fold_clip_probs,
        fold_metrics=[
            point_metrics(cohort.clip_labels, cp, thr)
            for cp, thr in zip(fold_clip_probs, clip_thresholds)
        ],
        fold_metrics_frame=[
            point_metrics(cohort.frame_labels, p, thr)
            for p, thr in zip(fold_probs, frame_thresholds)
        ],
        fold_thresholds=clip_thresholds,
        cv_mean_threshold=float(parent_metrics.get("cv_mean_threshold", 0.5)),
        cv_mean_clip_threshold=float(parent_metrics.get("cv_mean_clip_threshold", 0.5)),
    )
