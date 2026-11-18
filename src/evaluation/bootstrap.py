"""
src/evaluation/bootstrap.py
---------------------------
Patient-clustered bootstrap for the held-out analysis.

Resamples PATIENTS with replacement, never individual clips or frames, so
units from the same patient stay together. Statistics receive the draw counts
as a per-unit `sample_weight`, which lets one resample score several models at
once (paired differences keep the models' correlation).

Public API
----------
patient_clustered_bootstrap(labels, patient_ids, statistic_fn, n_bootstrap, seed, alpha)
    -> {point, replicates, n_dropped, ci_percentile, ci_bca}

metric_statistic(metric, labels, probs, threshold) -> statistic_fn
    One metric of one model. AUROC is threshold-free; the others use the fixed
    `threshold`, never re-tuned per resample.

delta_auroc_statistic(labels, probs_a, probs_b) -> statistic_fn
    AUROC(A) - AUROC(B), both scored on the same resampled weights.

flip_delta(point, ci) -> (-point, (-hi, -lo))
    Re-orient a difference as B - A. Exact for BCa, which is
    transformation-respecting (ties count as half in z0).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

Statistic = Callable[[np.ndarray], float]


# ---------------------------------------------------------------------------
# Bootstrap
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
                             point estimate, ties counted as half; acceleration from a
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
    # jackknife (Efron & Tibshirani, 1993, ch. 14). Replicates tied with the
    # point estimate count as half: AUROC-type statistics are discrete and
    # can pile up exactly on the point (e.g. a paired delta of exactly 0 when
    # both models reach AUROC 1.0 on a resample), and counting those ties as
    # "not below" would bias z0 negative and shift the whole interval down.
    prop_less = float(np.mean(replicates_arr < point) + 0.5 * np.mean(replicates_arr == point))
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
# Statistics
# ---------------------------------------------------------------------------


def metric_statistic(
    metric: str, labels: np.ndarray, probs: np.ndarray, threshold: float
) -> Statistic:
    """statistic_fn for one metric of one model, at clip or frame level alike.

    AUROC is threshold-free. Every other metric uses the fixed `threshold`
    (the one behind the point estimate), not re-optimized per resample, so the
    CI reflects patient sampling alone, not threshold instability.
    """
    preds = (probs >= threshold).astype(int)
    if metric == "auroc":
        return lambda w: float(roc_auc_score(labels, probs, sample_weight=w))
    if metric == "f1":
        return lambda w: float(f1_score(labels, preds, sample_weight=w, zero_division=0))
    if metric == "precision":
        return lambda w: float(precision_score(labels, preds, sample_weight=w, zero_division=0))
    if metric == "sensitivity":
        return lambda w: float(recall_score(labels, preds, sample_weight=w, zero_division=0))
    if metric == "accuracy":
        return lambda w: float(accuracy_score(labels, preds, sample_weight=w))
    if metric == "specificity":

        def _specificity(w):
            tn, fp, _, _ = confusion_matrix(labels, preds, labels=[0, 1], sample_weight=w).ravel()
            return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

        return _specificity
    raise ValueError(f"Unknown metric: {metric}")


def delta_auroc_statistic(
    labels: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray
) -> Statistic:
    """AUROC(A) - AUROC(B), both on the same resampled weights (paired)."""

    def _delta(w):
        return float(
            roc_auc_score(labels, probs_a, sample_weight=w)
            - roc_auc_score(labels, probs_b, sample_weight=w)
        )

    return _delta


def flip_delta(point: float, ci: tuple[float, float]) -> tuple[float, tuple[float, float]]:
    """Re-orient an A - B difference and its interval as B - A."""
    return -point, (-ci[1], -ci[0])
