"""
src/evaluation/metrics.py
-------------------------
Pure metric computation with bootstrap confidence intervals.

No I/O, no MLflow, no plotting, only numerical results.

Public API
----------
compute_metrics_with_ci(labels, preds, probs, n_bootstrap) -> dict
    Frame-level metrics with 95% bootstrap CIs.
    Returns keys "Metric" (formatted str) and "_Metric_mean" (float).

compute_clip_metrics(labels, probs, video_ids, threshold, aggregation_fn) -> dict
    Aggregates frame → clip then calls compute_metrics_with_ci.

bootstrap_ci(labels, preds, probs, metric_fn, n_bootstrap, ci, seed)
    -> (mean, lower, upper)

patient_clustered_bootstrap(labels, patient_ids, statistic_fn, n_bootstrap, seed, alpha)
    -> {point, replicates, n_dropped, ci_percentile, ci_bca}
    Cluster bootstrap over patients (resampling patients, not rows) for a
    statistic computed on clustered (e.g. clip-level) data. Use for a single
    metric (statistic_fn closes over one model's probs) or a paired
    difference (statistic_fn closes over two models' probs and returns
    their metric difference, evaluated on the SAME resampled weights).

fmt(mean, lower, upper) -> "0.876 (0.851–0.901)"
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt(mean: float, lower: float, upper: float) -> str:
    """Format a metric with its confidence interval: '0.876 (0.851–0.901)'."""
    if np.isnan(lower) or np.isnan(upper):
        return f"{mean} (CI unavailable)"
    return f"{mean} ({lower}–{upper})"


# ---------------------------------------------------------------------------
# Scalar metric functions  (uniform signature for bootstrap_ci)
# ---------------------------------------------------------------------------


def _auroc(labels, preds, probs):
    if probs.ndim == 2:
        probs = np.asarray(probs, dtype=float)
        present_classes = np.unique(labels)
        n_present = len(present_classes)

        if n_present < 2:
            return float("nan")  # cannot compute an AUROC

        if n_present < probs.shape[1]:
            print(
                f"  [warn] AUROC: {n_present}/{probs.shape[1]} classes present "
                f"({present_classes}), partial AUROC."
            )
            probs_partial = probs[:, present_classes]
            row_sums = probs_partial.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            probs_partial = probs_partial / row_sums

            if n_present == 2:
                # Degraded binary case, take the probability of the highest class
                return float(
                    roc_auc_score(
                        (labels == present_classes[1]).astype(int),
                        probs_partial[:, 1],
                    )
                )
            # 3+ classes but not all present
            return float(
                roc_auc_score(
                    labels,
                    probs_partial,
                    multi_class="ovr",
                    average="macro",
                    labels=present_classes,
                )
            )

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        probs = probs / row_sums
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))

    if len(np.unique(labels)) < 2:
        return float("nan")  # cannot compute an AUROC
    return float(roc_auc_score(labels, probs))


def _accuracy(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> float:
    return float(accuracy_score(labels, preds))


def _sensitivity(labels, preds, probs):
    avg = "binary" if probs.ndim == 1 else "macro"
    return float(recall_score(labels, preds, average=avg, zero_division=0))


def _precision(labels, preds, probs):
    avg = "binary" if probs.ndim == 1 else "macro"
    return float(precision_score(labels, preds, average=avg, zero_division=0))


def _f1(labels, preds, probs):
    avg = "binary" if probs.ndim == 1 else "macro"
    return float(f1_score(labels, preds, average=avg, zero_division=0))


def _specificity(labels, preds, probs):
    if probs.ndim == 2:
        return float("nan")  # not defined in multiclass
    tn, fp, _, _ = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


METRIC_FNS: dict[str, Callable] = {
    "Accuracy": _accuracy,
    "AUROC": _auroc,
    "Sensitivity": _sensitivity,
    "Specificity": _specificity,
    "Precision": _precision,
    "F1": _f1,
}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci: int = 95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean ± bootstrap CI for a single metric.

    Args:
        labels:      Ground-truth binary labels.
        preds:       Binary predictions.
        probs:       Predicted probabilities for the positive class.
        metric_fn:   Callable with signature (labels, preds, probs) -> float.
        n_bootstrap: Number of resampling iterations.
        ci:          Confidence level in % (default 95).
        seed:        Random seed for reproducibility.

    Returns:
        (mean, lower, upper) rounded to 3 decimal places.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        min_classes = 2
        if len(np.unique(labels[idx])) < min_classes:
            continue
        scores.append(metric_fn(labels[idx], preds[idx], probs[idx]))

    if not scores:
        point_estimate = round(float(metric_fn(labels, preds, probs)), 3)
        print("  [warn] bootstrap_ci: no valid samples, CI unavailable, returning point estimate.")
        return point_estimate, float("nan"), float("nan")

    alpha = (100 - ci) / 2
    return (
        round(float(np.mean(scores)), 3),
        round(float(np.percentile(scores, alpha)), 3),
        round(float(np.percentile(scores, 100 - alpha)), 3),
    )


def bootstrap_ci_aggregated(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n: int = 10_000,
    ci: float = 95,
    seed: int = 42,
) -> tuple[float, float]:
    """Unified bootstrap CI for aggregated metrics.

    Used for clip-level or patient-level aggregated metrics where we need
    to resample from the aggregated data rather than frame-level data.

    Args:
        y_true: Ground-truth labels (aggregated level).
        y_score: Predicted scores (aggregated level).
        metric_fn: Metric function callable(y_true, y_score) -> float.
        n: Number of bootstrap resamples.
        ci: Confidence interval level (default 95%).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) at specified confidence level.
    """
    rng = np.random.default_rng(seed)
    scores, size = [], len(y_true)
    for _ in range(n):
        idx = rng.integers(0, size, size=size)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue

    if not scores:
        return float("nan"), float("nan")

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper


# ---------------------------------------------------------------------------
# Patient-clustered bootstrap
# ---------------------------------------------------------------------------


def patient_clustered_bootstrap(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Cluster bootstrap over patients for a statistic on clustered units.

    Resamples PATIENTS with replacement (not the clustered units themselves,
    e.g. clips), so units sharing a patient are resampled together — this is
    the correction for within-patient correlation that a plain row-wise
    bootstrap (bootstrap_ci_aggregated) ignores.

    `statistic_fn` takes a per-unit `sample_weight` array (draw counts, 0 for
    unsampled units) and returns a scalar, e.g.
        lambda w: roc_auc_score(labels, probs, sample_weight=w)
    for a single-model metric, or
        lambda w: (roc_auc_score(labels, probs_a, sample_weight=w)
                   - roc_auc_score(labels, probs_b, sample_weight=w))
    for a paired difference — both models are evaluated on the SAME
    resampled weights each iteration, preserving their correlation (the
    same role DeLong's covariance term plays, but done via resampling).

    Args:
        labels:       Ground-truth labels for each clustered unit (e.g. one
                      per clip), shape (n_units,).
        patient_ids:  Patient/cluster identifier for each unit, shape (n_units,).
        statistic_fn: Callable(sample_weight: np.ndarray) -> float.
        n_bootstrap:  Number of bootstrap iterations.
        seed:         Random seed for reproducibility.
        alpha:        Two-sided significance level (default 0.05 -> 95% CI).

    Returns:
        dict with:
          "point"         -> float, plug-in statistic on the full (unresampled)
                             data, i.e. statistic_fn(ones). This is the
                             reported point estimate, NOT the bootstrap
                             mean/median (the bootstrap distribution
                             characterizes uncertainty around the point
                             estimate, it isn't a better estimate of it).
          "replicates"    -> np.ndarray of the (n_bootstrap - n_dropped) valid
                             bootstrap replicates.
          "n_dropped"     -> int, replicates skipped because the resampled
                             label set was single-class (statistic undefined).
          "ci_percentile" -> (lower, upper), plain percentile CI.
          "ci_bca"        -> (lower, upper), bias-corrected accelerated CI
                             (bias from the fraction of replicates below the
                             point estimate; acceleration from a
                             leave-one-patient-out jackknife). More robust
                             than percentile under skew with few clusters.
    """
    labels = np.asarray(labels)
    patient_ids = np.asarray(patient_ids)
    n_units = len(labels)

    unique_patients, clip_patient_idx = np.unique(patient_ids, return_inverse=True)
    n_patients = len(unique_patients)

    point = float(statistic_fn(np.ones(n_units, dtype=float)))

    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    n_dropped = 0
    for _ in range(n_bootstrap):
        draw_counts = np.bincount(
            rng.integers(0, n_patients, size=n_patients), minlength=n_patients
        )
        weight = draw_counts[clip_patient_idx].astype(float)
        if len(np.unique(labels[weight > 0])) < 2:
            n_dropped += 1
            continue
        replicates.append(float(statistic_fn(weight)))

    replicates_arr = np.array(replicates, dtype=float)
    nan_ci = (float("nan"), float("nan"))
    if replicates_arr.size == 0:
        return {
            "point": point,
            "replicates": replicates_arr,
            "n_dropped": n_dropped,
            "ci_percentile": nan_ci,
            "ci_bca": nan_ci,
        }

    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)
    ci_percentile = (
        float(np.percentile(replicates_arr, lo_pct)),
        float(np.percentile(replicates_arr, hi_pct)),
    )

    # BCa: bias correction z0 from the fraction of replicates below the
    # plug-in point estimate; acceleration a from a leave-one-patient-out
    # jackknife (Efron & Tibshirani, 1993, ch. 14).
    prop_less = float(np.mean(replicates_arr < point))
    z0 = norm.ppf(np.clip(prop_less, 1e-6, 1 - 1e-6))

    jack_vals: list[float] = []
    for p_idx in range(n_patients):
        weight = np.where(clip_patient_idx == p_idx, 0.0, 1.0)
        if len(np.unique(labels[weight > 0])) < 2:
            continue  # degenerate leave-one-out, drop from the jackknife
        jack_vals.append(float(statistic_fn(weight)))
    jack_arr = np.array(jack_vals, dtype=float)

    if jack_arr.size < 2:
        a_hat = 0.0  # not enough jackknife points to estimate acceleration
    else:
        jack_mean = jack_arr.mean()
        num = np.sum((jack_mean - jack_arr) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack_arr) ** 2) ** 1.5)
        a_hat = float(num / den) if den != 0 else 0.0

    def _bca_percentile(z_conf: float) -> float:
        adj_num = z0 + z_conf
        denom = 1 - a_hat * adj_num
        adj = z0 + adj_num / denom
        return float(np.clip(norm.cdf(adj) * 100, 0.0, 100.0))

    p_lo = _bca_percentile(norm.ppf(alpha / 2))
    p_hi = _bca_percentile(norm.ppf(1 - alpha / 2))
    ci_bca = (
        float(np.percentile(replicates_arr, p_lo)),
        float(np.percentile(replicates_arr, p_hi)),
    )

    return {
        "point": point,
        "replicates": replicates_arr,
        "n_dropped": n_dropped,
        "ci_percentile": ci_percentile,
        "ci_bca": ci_bca,
    }


# ---------------------------------------------------------------------------
# Frame-level metrics with CI
# ---------------------------------------------------------------------------


def compute_metrics_with_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute all metrics in METRIC_FNS with bootstrap CIs.

    Returns:
        Dict with:
          - "Metric"        -> str  formatted as "mean (lower–upper)"
          - "_Metric_mean"  -> float  raw mean (for plots and comparisons)
    """
    result: dict = {}
    for name, fn in METRIC_FNS.items():
        mean, lower, upper = bootstrap_ci(labels, preds, probs, fn, n_bootstrap, seed=seed)
        if np.isnan(mean):
            continue  # ← Specificity silently skipped in multiclass
        result[name] = fmt(mean, lower, upper)
        result[f"_{name}_mean"] = mean
        result[f"_{name}_lower"] = lower
        result[f"_{name}_upper"] = upper
    return result


# ---------------------------------------------------------------------------
# Clip-level metrics with CI
# ---------------------------------------------------------------------------


def compute_clip_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    video_ids: list[str],
    threshold: float = 0.5,
    aggregation_fn: Callable[[np.ndarray], float] = np.mean,
) -> dict:
    """Aggregate frame predictions to clip level, then compute metrics.

    Args:
        labels:         Frame-level ground-truth labels (0/1).
        probs:          Frame-level predicted probabilities P(ulcer).
        video_ids:      Clip identifier for each frame.
        threshold:      Decision threshold on the aggregated probability.
        aggregation_fn: Aggregation function for probabilities (default: mean).
                        For the full set of 12 methods, use aggregation.py.

    Returns:
        Same format as compute_metrics_with_ci.
    """
    df = pd.DataFrame({"video_id": video_ids, "label": labels, "prob": probs})

    # Use pandas’ string aggregation name for mean to avoid FutureWarning on callables
    agg_fn = "mean" if aggregation_fn is np.mean else aggregation_fn

    clip_df = (
        df.groupby("video_id").agg(label=("label", "max"), prob=("prob", agg_fn)).reset_index()
    )

    clip_labels = clip_df["label"].to_numpy(dtype=int)
    clip_probs = clip_df["prob"].to_numpy(dtype=float)
    clip_preds = (clip_probs >= threshold).astype(int)

    return compute_metrics_with_ci(clip_labels, clip_preds, clip_probs)
