"""
src/evaluation/cv_tables.py
---------------------------
Tables of the cross-validation (validation-fold) report, from the metrics and
artifacts that run_modes.py logged during CV training. No re-inference.

Public API
----------
main_table(runs) -> one row per configuration (parent-run CV aggregates)
per_fold_table(runs, with_val_specificity) -> (per-fold DataFrame, diagnostics)
fold_auroc_matrix(fold_df, order) -> folds x models AUROC, with Mean / Std
"""

from __future__ import annotations

import os
from typing import NamedTuple

import mlflow
import numpy as np
import pandas as pd

from src.config.models import get_model_entry
from src.evaluation.metrics import METRIC_FNS
from src.evaluation.mlflow_io import CVRuns

# Fallback when a fold logged no tuned threshold
CV_THRESHOLD = 0.5

# Parent-logged CV aggregates
VAL_METRICS = ("auroc", "f1", "precision", "recall")

# Held-out per-fold metrics logged on each CV child run
_REPORT_METRICS = ("auroc", "sensitivity", "specificity", "f1")
_HELDOUT_LEVELS = {
    "frame": "heldout__{metric}_mean",
    "clip": "heldout_clip__{metric}_mean",
}

# Recomputed from the val-fold probability artifacts; AUROC and F1 are already
# logged and only recomputed to cross-check each fold against its logged values.
_VAL_ARTIFACT_METRICS = ("AUROC", "Sensitivity", "Specificity", "F1")

# Logged metrics are full float64, so exact agreement is expected; 1e-4 only
# absorbs float round-trips.
_CROSSCHECK_TOL = 1e-4


def _parent_metric(parent, key: str, default: float = float("nan")) -> float:
    return float(parent.data.metrics.get(key, default))


def _parent_param(parent, key: str, default=None):
    return parent.data.params.get(key, default)


def main_table(runs: dict[str, CVRuns]) -> pd.DataFrame:
    rows = []
    for model_key, (parent, children) in runs.items():
        entry = get_model_entry(model_key)
        row = {
            "config": model_key,
            "architecture": entry.architecture,
            "pretrain_data": entry.pretrain_data,
            "pretrain_method": entry.pretrain_method,
            "n_folds": len(children),
        }
        for m in VAL_METRICS:
            row[f"val_{m}_mean"] = round(_parent_metric(parent, f"cv_mean_val_{m}"), 4)
            row[f"val_{m}_std"] = round(_parent_metric(parent, f"cv_std_val_{m}"), 4)
        for stat in ("mean", "std", "min", "max"):
            row[f"threshold_{stat}"] = round(_parent_metric(parent, f"cv_{stat}_threshold"), 4)

        best_fold = _parent_param(parent, "best_fold")
        row["best_fold"] = int(best_fold) + 1 if best_fold is not None else None
        best_auroc = _parent_param(parent, "best_fold_val_auroc")
        row["best_fold_val_auroc"] = round(float(best_auroc), 4) if best_auroc else float("nan")
        best_thr = _parent_param(parent, "best_fold_threshold")
        row["best_fold_threshold"] = round(float(best_thr), 4) if best_thr else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _val_fold_metrics_from_artifacts(child, fold: int) -> dict | None:
    """Recompute validation-fold metrics from that fold's probability artifacts.

    Returns {"auroc","sensitivity","specificity","f1", "_mismatch": [...]} or
    None when the artifacts are missing/unreadable (one fold in this project
    has a 0-byte val_probs file, so this must degrade rather than raise).

    The threshold used is the fold's own tuned threshold, matching what
    run_modes.py applied when it logged fold_val_{f1,precision,recall}.
    """
    uri = f"{child.info.artifact_uri}/predictions/cv_folds"
    try:
        local = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        probs = np.load(os.path.join(local, f"val_probs_fold{fold}.npy"))
        labels = np.load(os.path.join(local, f"val_labels_fold{fold}.npy"))
    except Exception:
        return None
    if probs.size == 0 or labels.size == 0 or probs.shape[0] != labels.shape[0]:
        return None

    if probs.ndim == 2:
        probs = probs[:, 1]
    threshold = child.data.metrics.get("fold_optimal_threshold", CV_THRESHOLD)
    preds = (probs >= threshold).astype(int)

    values = {
        name.lower(): METRIC_FNS[name](labels, preds, probs) for name in _VAL_ARTIFACT_METRICS
    }

    mismatch = []
    for name, logged_key in (
        ("sensitivity", "fold_val_recall"),
        ("f1", "fold_val_f1"),
        ("auroc", "fold_val_auroc"),
    ):
        logged = child.data.metrics.get(logged_key)
        if logged is not None and abs(values[name] - logged) > _CROSSCHECK_TOL:
            mismatch.append(f"{name}: recomputed {values[name]:.4f} vs logged {logged:.4f}")
    values["_mismatch"] = mismatch
    return values


class FoldDiagnostics(NamedTuple):
    artifact_misses: list[str]  # folds without a readable val probability artifact
    crosscheck_failures: list[str]  # folds whose recomputed metrics disagree with MLflow


def per_fold_table(
    runs: dict[str, CVRuns], with_val_specificity: bool = True
) -> tuple[pd.DataFrame, FoldDiagnostics]:
    """Per-fold validation metrics (logged), val sensitivity/specificity recomputed
    from the probability artifacts, and the held-out metrics logged per fold."""
    rows, misses, failures = [], [], []
    for model_key, (_, children) in runs.items():
        entry = get_model_entry(model_key)
        for child in children:
            metrics = child.data.metrics
            fold_idx = int(child.data.params.get("fold", -1))
            row = {
                "config": model_key,
                "architecture": entry.architecture,
                "fold": fold_idx + 1,
                "val_auroc": metrics.get("fold_val_auroc"),
                "val_f1": metrics.get("fold_val_f1"),
                "val_precision": metrics.get("fold_val_precision"),
                "val_recall": metrics.get("fold_val_recall"),
                "optimal_threshold": metrics.get("fold_optimal_threshold"),
                "val_sensitivity": metrics.get("fold_val_recall"),
                "val_specificity": None,
            }
            if with_val_specificity:
                recomputed = _val_fold_metrics_from_artifacts(child, fold_idx)
                if recomputed is None:
                    misses.append(f"{model_key} fold {fold_idx + 1}")
                else:
                    row["val_sensitivity"] = recomputed["sensitivity"]
                    row["val_specificity"] = recomputed["specificity"]
                    failures += [
                        f"{model_key} fold {fold_idx + 1}: {msg}" for msg in recomputed["_mismatch"]
                    ]
            for level, key_template in _HELDOUT_LEVELS.items():
                for m in _REPORT_METRICS:
                    row[f"heldout_{level}_{m}"] = metrics.get(key_template.format(metric=m))
            rows.append(row)
    return pd.DataFrame(rows), FoldDiagnostics(misses, failures)


def fold_auroc_matrix(fold_auroc_wide: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Folds as columns (fold_k), Mean / Std across folds, rows in `order`."""
    matrix = fold_auroc_wide.copy()
    matrix.columns = [f"fold_{c}" for c in matrix.columns]
    matrix["Mean"] = fold_auroc_wide.mean(axis=1)
    matrix["Std"] = fold_auroc_wide.std(axis=1, ddof=1)
    return matrix.reindex(order)
