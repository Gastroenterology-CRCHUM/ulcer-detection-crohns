"""
scripts/ulcer/effect_decomposition.py
--------------------------------------
Heldout-cohort 5-fold ensemble evaluation and pretraining effect decomposition.

For each of the 9 trained model configurations, averages the 5 CV folds'
already-logged heldout frame probabilities (no retraining, no re-inference)
into a single ensemble prediction per config, aggregates to clip level, and:

  1. Reports each config's ensemble AUROC (and 5 other metrics) with a
     patient-clustered bootstrap CI (resampling the 19 heldout patients),
     at BOTH clip level (primary -- the grain the pairwise tests below use)
     and frame level (descriptive companion, same patient-level resampling),
     plus ROC curves and rank tests at both levels.
  2. Decomposes two pretraining effects across 5 architecture-matched pairs:
       method effect (Supervised -> DINOv1, ImageNet held constant)
       corpus effect (ImageNet -> GastroNet-5M, DINOv1 held constant)
     reporting, per pair: the plug-in ensemble DeltaAUROC with a patient-
     clustered bootstrap CI (primary uncertainty statement) alongside
     ensemble DeLong and best-fold DeLong (both clip-level, non-clustered,
     kept for traceability against the clustered result).

Usage
-----
    python scripts/ulcer/effect_decomposition.py
    python scripts/ulcer/effect_decomposition.py --dry-run
    python scripts/ulcer/effect_decomposition.py --n-bootstrap 20000 --seed 7
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mlflow
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
    roc_curve,
)

from src.config.models import MODEL_REGISTRY, get_model_entry
from src.config.paths import get_default_paths
from src.evaluation.delong import delong_test
from src.evaluation.metrics import patient_clustered_bootstrap
from src.evaluation.mlflow_io import download_npy
from src.evaluation.plots import plot_roc_curves
from src.evaluation.rank_tests import (
    holm_bonferroni,
    plot_friedman_ranks,
    run_friedman,
)
from src.evaluation.reporting import to_markdown
from src.evaluation.style import (
    STATUS_NEUTRAL,
    STATUS_NOT_SIGNIFICANT,
    STATUS_SIGNIFICANT,
    config_color,
    config_label,
    config_labels,
    level_label,
    metric_label,
    slug,
)

plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# Limitations (documentation only -- not printed at runtime; kept here for
# whoever reads this analysis code / writes the manuscript's methods section)
# ---------------------------------------------------------------------------
#
# - Patient-clustered bootstrap CIs resample patients but evaluate every
#   non-AUROC metric at a FIXED threshold (the point estimate's own
#   cv_mean_clip_threshold) -- they do not re-tune the threshold per
#   resample, so the CI reflects patient-cohort sampling variability alone,
#   not threshold instability.
# - Ensemble/best-fold DeLong (both clip-level) are kept for traceability
#   against the primary patient-clustered bootstrap CI, not as a second
#   significance claim -- they treat clips as independent, which
#   pseudo-replicates clips from the same patient.
# - The full 9x9 pairwise bootstrap matrix is exploratory/traceability
#   only: it is not corrected for multiple comparisons, and only 5 of its
#   36 pairs are the pre-specified architecture-matched comparisons this
#   script's primary claims rest on.
# - Frame-level results throughout are descriptive companions, not a
#   second significance claim -- clip-level is the grain the pairwise
#   effect-decomposition tests are run on. The per-config ensemble-metrics
#   table's frame-level 95% CI uses the SAME patient-clustered resampling
#   as clip-level (resampling patients, not frames), so it is not a naive
#   per-frame bootstrap; the per-fold dispersion table's frame-level values
#   remain a simple mean +/- std across the 5 individually-trained fold
#   models, not a bootstrap CI at all.
# - When a parent run's predictions/heldout_best_fold_probs.npy artifact is
#   missing or shape-mismatched, best_fold_clip_probs silently falls back
#   to the ensemble prediction for that config (see the printed [warn] at
#   load time) -- best-fold DeLong then compares the ensemble to itself.

# ---------------------------------------------------------------------------
# Comparison pairs (hardcoded -- see module docstring / plan for rationale)
# ---------------------------------------------------------------------------

PAIRS: list[dict] = [
    {
        "pair_id": "method_resnet50",
        "effect": "method",
        "architecture": "ResNet-50",
        "config_a": "resnet50_imagenet_sup",
        "config_b": "resnet50_imagenet",
    },
    {
        "pair_id": "method_vitb16",
        "effect": "method",
        "architecture": "ViT-Base/16",
        "config_a": "vitb16_imagenet_sup",
        "config_b": "vitb16_imagenet",
    },
    {
        "pair_id": "method_vits16",
        "effect": "method",
        "architecture": "ViT-Small/16",
        "config_a": "vits16_imagenet_hf",
        "config_b": "vits16_imagenet",
    },
    {
        "pair_id": "corpus_resnet50",
        "effect": "corpus",
        "architecture": "ResNet-50",
        "config_a": "resnet50_imagenet",
        "config_b": "resnet50_gastronet",
    },
    {
        "pair_id": "corpus_vits16",
        "effect": "corpus",
        "architecture": "ViT-Small/16",
        "config_a": "vits16_imagenet",
        "config_b": "vits16_gastronet",
    },
]

# ---------------------------------------------------------------------------
# Clip-level aggregation
# ---------------------------------------------------------------------------


def _aggregate_to_clip(
    frame_probs: np.ndarray, clip_keys: np.ndarray, clip_order: list
) -> np.ndarray:
    """Mean-aggregate frame probabilities to clip level, in a fixed clip order."""
    df = pd.DataFrame({"clip_key": clip_keys, "prob": frame_probs})
    clip_means = df.groupby("clip_key", sort=False)["prob"].mean()
    return clip_means.reindex(clip_order).to_numpy()


def _clip_point_metrics(clip_probs: np.ndarray, clip_labels: np.ndarray, threshold: float) -> dict:
    """Point (non-bootstrapped) clip-level metrics at a fixed threshold."""
    preds = (clip_probs >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(clip_labels, preds, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auroc = (
        roc_auc_score(clip_labels, clip_probs) if len(np.unique(clip_labels)) > 1 else float("nan")
    )
    return {
        "auroc": auroc,
        "accuracy": accuracy_score(clip_labels, preds),
        "f1": f1_score(clip_labels, preds, zero_division=0),
        "precision": precision_score(clip_labels, preds, zero_division=0),
        "sensitivity": recall_score(clip_labels, preds, zero_division=0),
        "specificity": spec,
    }


_METRIC_ORDER = ("auroc", "f1", "precision", "sensitivity", "specificity", "accuracy")


def _make_metric_bootstrap_fn(
    metric_name: str, labels: np.ndarray, probs: np.ndarray, threshold: float
):
    """statistic_fn factory for patient_clustered_bootstrap. Level-agnostic --
    the caller passes clip-level or frame-level (labels, probs) and the same
    logic applies, since patient_clustered_bootstrap only cares that
    `labels`/the array `probs`/`preds` line up one-to-one with `patient_ids`.

    AUROC is threshold-free (probability-based). Every other metric is
    evaluated at the FIXED `threshold` (the same one used for the point
    estimate, e.g. cv_mean_clip_threshold) -- not re-optimized per bootstrap
    resample, so the CI reflects sampling variability of the patient cohort
    alone, not threshold instability.
    """
    preds = (probs >= threshold).astype(int)
    if metric_name == "auroc":
        return lambda w: float(roc_auc_score(labels, probs, sample_weight=w))
    if metric_name == "f1":
        return lambda w: float(f1_score(labels, preds, sample_weight=w, zero_division=0))
    if metric_name == "precision":
        return lambda w: float(precision_score(labels, preds, sample_weight=w, zero_division=0))
    if metric_name == "sensitivity":
        return lambda w: float(recall_score(labels, preds, sample_weight=w, zero_division=0))
    if metric_name == "accuracy":
        return lambda w: float(accuracy_score(labels, preds, sample_weight=w))
    if metric_name == "specificity":

        def _specificity_fn(w):
            tn, fp, _, _ = confusion_matrix(labels, preds, labels=[0, 1], sample_weight=w).ravel()
            return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

        return _specificity_fn
    raise ValueError(f"Unknown metric: {metric_name}")


# ---------------------------------------------------------------------------
# Per-config data loading
# ---------------------------------------------------------------------------


def _load_config(
    client: MlflowClient,
    parent,
    children: list,
    clip_order: list,
    clip_keys_frame: np.ndarray,
    clip_labels: np.ndarray,
    manifest_frame_labels: np.ndarray,
) -> dict | None:
    """Download and aggregate one config's heldout predictions.

    Returns None (with a printed warning) if any expected artifact is
    missing or malformed, so the caller can skip that config.

    Clip aggregation assumes each fold's `heldout_probs_fold{k}.npy` is in
    the SAME row order as the heldout manifest (so `probs[i]` corresponds to
    `clip_keys_frame[i]`) -- this is verified per fold against that fold's
    own `heldout_labels_fold{k}.npy` sibling artifact rather than assumed.
    """
    fold_frame_probs, fold_thresholds, fold_frame_thresholds = [], [], []
    for ch in children:
        fold_name = ch.data.tags.get("mlflow.runName", ch.info.run_id[:8])
        arts = []
        try:
            arts = [a.path for a in client.list_artifacts(ch.info.run_id, "predictions/heldout")]
        except Exception:
            pass
        probs_paths = [a for a in arts if "probs" in a]
        labels_paths = [a for a in arts if "labels" in a]
        if not probs_paths or not labels_paths:
            print(
                f"    [warn] {fold_name}: missing heldout probs/labels artifact, skipping config."
            )
            return None
        probs = download_npy(client, ch.info.run_id, probs_paths[0])
        fold_labels = download_npy(client, ch.info.run_id, labels_paths[0])
        if len(probs) != len(clip_keys_frame) or len(fold_labels) != len(clip_keys_frame):
            print(
                f"    [warn] {fold_name}: {len(probs)} probs / {len(fold_labels)} labels, expected "
                f"{len(clip_keys_frame)}, skipping config."
            )
            return None
        if not np.array_equal(fold_labels, manifest_frame_labels):
            print(
                f"    [warn] {fold_name}: heldout labels do not match the manifest row order "
                f"(frame/clip alignment would be wrong), skipping config."
            )
            return None
        fold_frame_probs.append(probs)
        fold_thresholds.append(float(ch.data.metrics.get("fold_optimal_clip_threshold", 0.5)))
        fold_frame_thresholds.append(float(ch.data.metrics.get("fold_optimal_threshold", 0.5)))

    fold_clip_probs = [_aggregate_to_clip(p, clip_keys_frame, clip_order) for p in fold_frame_probs]
    fold_metrics = [
        _clip_point_metrics(cp, clip_labels, thr)
        for cp, thr in zip(fold_clip_probs, fold_thresholds)
    ]
    fold_metrics_frame = [
        _clip_point_metrics(p, manifest_frame_labels, thr)
        for p, thr in zip(fold_frame_probs, fold_frame_thresholds)
    ]

    ensemble_frame_probs = np.mean(np.stack(fold_frame_probs, axis=0), axis=0)
    ensemble_clip_probs = _aggregate_to_clip(ensemble_frame_probs, clip_keys_frame, clip_order)

    try:
        best_fold_frame_probs = download_npy(
            client, parent.info.run_id, "predictions/heldout_best_fold_probs.npy"
        )
        best_fold_frame_labels = download_npy(
            client, parent.info.run_id, "predictions/heldout_best_fold_labels.npy"
        )
        if len(best_fold_frame_probs) != len(clip_keys_frame) or not np.array_equal(
            best_fold_frame_labels, manifest_frame_labels
        ):
            raise ValueError("best-fold heldout artifact shape/order mismatch")
        best_fold_clip_probs = _aggregate_to_clip(
            best_fold_frame_probs, clip_keys_frame, clip_order
        )
    except Exception as exc:
        print(
            f"    [warn] best-fold heldout probs unavailable ({exc}), using ensemble as fallback."
        )
        best_fold_clip_probs = ensemble_clip_probs

    cv_mean_clip_threshold = float(parent.data.metrics.get("cv_mean_clip_threshold", 0.5))
    cv_mean_threshold = float(parent.data.metrics.get("cv_mean_threshold", 0.5))

    return {
        "ensemble_frame_probs": ensemble_frame_probs,
        "ensemble_clip_probs": ensemble_clip_probs,
        "best_fold_clip_probs": best_fold_clip_probs,
        "fold_clip_probs": fold_clip_probs,
        "fold_metrics": fold_metrics,
        "fold_metrics_frame": fold_metrics_frame,
        "fold_thresholds": fold_thresholds,
        "cv_mean_threshold": cv_mean_threshold,
        "cv_mean_clip_threshold": cv_mean_clip_threshold,
    }


def _load_all_configs(
    client: MlflowClient,
    exp_id: str,
    clip_order: list,
    clip_keys_frame: np.ndarray,
    clip_labels: np.ndarray,
    manifest_frame_labels: np.ndarray,
) -> dict:
    all_runs = client.search_runs(experiment_ids=[exp_id], max_results=1000)
    parents_by_model = {
        r.data.tags.get("model"): r for r in all_runs if not r.data.tags.get("mlflow.parentRunId")
    }
    children_by_parent: dict = {}
    for r in all_runs:
        pid = r.data.tags.get("mlflow.parentRunId")
        if pid:
            children_by_parent.setdefault(pid, []).append(r)

    configs: dict = {}
    for model_key in MODEL_REGISTRY:
        parent = parents_by_model.get(model_key)
        if parent is None:
            print(f"  [warn] {model_key}: no parent run found, skipping.")
            continue
        # Sort by the actual "fold" param, not the runName string (a
        # lexicographic sort over e.g. "fold_1".."fold_10" would not even
        # match numeric fold order past 9 folds, and it never guaranteed
        # per_fold_heldout_metrics.csv's "fold" column meant the same fold
        # as cv_results.py's, which sorts on this same int param).
        children = sorted(
            children_by_parent.get(parent.info.run_id, []),
            key=lambda r: int(r.data.params.get("fold", -1)),
        )
        if not children:
            print(f"  [warn] {model_key}: no child fold runs, skipping.")
            continue

        print(f"  {model_key}: loading {len(children)} fold(s)...")
        data = _load_config(
            client,
            parent,
            children,
            clip_order,
            clip_keys_frame,
            clip_labels,
            manifest_frame_labels,
        )
        if data is not None:
            configs[model_key] = data
    return configs


# ---------------------------------------------------------------------------
# Plotting -- bar/forest charts
# ---------------------------------------------------------------------------


def _plot_bar_with_ci(names, points, cis, title, ylabel, output_path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.0), 5))
    x = np.arange(len(names))
    lo = np.array([max(points[i] - cis[i][0], 0.0) for i in range(len(names))])
    hi = np.array([max(cis[i][1] - points[i], 0.0) for i in range(len(names))])
    colors = [config_color(n) for n in names]
    ax.bar(x, points, yerr=[lo, hi], capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels(names), rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_forest(
    pair_df: pd.DataFrame, ci_lo_col: str, ci_hi_col: str, title: str, output_path
) -> None:
    """One row per comparison pair. Point = plug-in ensemble DeltaAUROC (A - B), whiskers =
    the bootstrap CI named by ci_lo_col/ci_hi_col. Color marks whether that CI excludes
    zero (the primary significance call). DeLong significance (ensemble/best-fold, kept
    for traceability, see module docstring) is annotated alongside each point rather than
    driving the color, since it is not the primary test."""
    n = len(pair_df)
    fig, ax = plt.subplots(figsize=(9.5, max(3.5, n * 1.1)))

    labels = []
    max_hi = pair_df[ci_hi_col].max()
    for i, row in enumerate(pair_df.itertuples()):
        pt = row.delta_auroc_ensemble
        lo, hi = getattr(row, ci_lo_col), getattr(row, ci_hi_col)
        significant = lo > 0 or hi < 0
        color = STATUS_SIGNIFICANT if significant else STATUS_NEUTRAL
        ax.errorbar(
            pt,
            i,
            xerr=[[max(pt - lo, 0.0)], [max(hi - pt, 0.0)]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=8,
            zorder=3,
        )
        marks = []
        if row.delong_ensemble_p < 0.05:
            marks.append("DeLong-ens*")
        if row.delong_bestfold_p < 0.05:
            marks.append("DeLong-best*")
        if marks:
            ax.text(
                max_hi + 0.01, i, " ".join(marks), va="center", fontsize=7.5, color=STATUS_NEUTRAL
            )
        labels.append(
            f"{row.pair_id}\nA: {config_label(row.config_a)}\nB: {config_label(row.config_b)}"
        )

    ax.axvline(0, linestyle="--", color="#c3c2b7", linewidth=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("DeltaAUROC  (A - B)")
    ax.set_title(title)
    legend_elems = [
        mpatches.Patch(facecolor=STATUS_SIGNIFICANT, label="Bootstrap CI excludes 0 (significant)"),
        mpatches.Patch(facecolor=STATUS_NEUTRAL, label="Bootstrap CI includes 0 (not significant)"),
    ]
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    ax.legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        fontsize=8,
        frameon=True,
    )
    fig.text(
        0.02,
        0.02,
        "DeLong-ens*/DeLong-best* = ensemble/best-fold DeLong p<0.05 (non-clustered, kept for "
        "traceability -- see the bootstrap CI for the primary significance call).",
        fontsize=7,
        color=STATUS_NEUTRAL,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_grouped_metrics(ax, names, metric_values: dict, ci_values=None, thresholds=None) -> None:
    """Draw one grouped-metrics panel onto `ax` -- metrics on the x-axis, one
    bar per model per metric group. No figure/save/legend here: the caller
    (currently only the two-level frame-above/clip-below wrapper, which
    shares one legend across both panels) owns legend placement.

    ci_values, if given, maps metric name -> list of (lo, hi) or None per
    model, for metrics that actually have a computed CI -- metrics without
    one are plotted as plain point bars, no CI is fabricated for them.
    thresholds, if given, maps model name -> decision threshold, shown in
    each bar's legend label since every non-AUROC metric here is evaluated
    at that fixed threshold."""
    metric_names = list(metric_values.keys())
    n_models = len(names)
    x = np.arange(len(metric_names))
    width = 0.8 / n_models
    for i, name in enumerate(names):
        values = [metric_values[m][i] for m in metric_names]
        yerr = None
        if ci_values is not None:
            lo_list, hi_list, has_any = [], [], False
            for m in metric_names:
                ci = ci_values.get(m, [None] * n_models)[i]
                if ci is not None:
                    has_any = True
                    lo_list.append(max(metric_values[m][i] - ci[0], 0.0))
                    hi_list.append(max(ci[1] - metric_values[m][i], 0.0))
                else:
                    lo_list.append(0.0)
                    hi_list.append(0.0)
            yerr = [lo_list, hi_list] if has_any else None
        display_name = config_label(name)
        label = display_name if thresholds is None else f"{display_name}  (thr={thresholds[name]:.2f})"
        ax.bar(
            x + i * width,
            values,
            width,
            label=label,
            color=config_color(name),
            yerr=yerr,
            capsize=3,
        )
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([metric_label(m) for m in metric_names], rotation=0)
    ax.set_ylim(0, 1.05)




def _plot_two_level_grouped_metrics(
    names, frame_values: dict, frame_ci: dict, clip_values: dict, clip_ci: dict, title, output_path
) -> None:
    """Frame level on top, clip level on the bottom, ONE shared legend
    (model name only -- no per-model threshold in this merged figure: frame
    and clip thresholds differ per model, so a single shared legend cannot
    show both without either duplicating or conflicting; the per-model
    threshold values are in heldout_ensemble_metrics.csv's threshold /
    threshold_frame columns, and a footnote states what each level's
    threshold IS instead). Both panels now carry a 95% percentile bootstrap
    CI: frame-level uses the SAME patient-clustered resampling as clip-level
    (resampling the 19 heldout patients, evaluating the metric over that
    patient subset's frames each time) -- not a naive per-frame bootstrap,
    so it does not reintroduce the frame-count pseudo-replication a plain
    row-wise resample would."""
    fig, axes = plt.subplots(
        2, 1, figsize=(max(11, len(_METRIC_ORDER) * 2.2), 11.6), sharex=True, sharey=True
    )
    _draw_grouped_metrics(axes[0], names, frame_values, frame_ci)
    axes[0].set_title("Frame level (95% CI)", fontsize=11, fontweight="bold")
    _draw_grouped_metrics(axes[1], names, clip_values, clip_ci)
    axes[1].set_title("Clip level, mean-probability aggregation (95% CI)", fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=True)
    fig.text(
        0.5,
        0.02,
        "Threshold applied at each level = cv_mean_threshold (frame) / cv_mean_clip_threshold (clip) "
        "-- the CV-averaged per-fold optimal threshold; see heldout_ensemble_metrics.csv for per-model values.",
        ha="center",
        fontsize=7.5,
        color=STATUS_NEUTRAL,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_bootstrap_matrix(
    p_matrix_boot: pd.DataFrame, bootstrap_matrix_df: pd.DataFrame, alpha: float, output_path
) -> None:
    """Full pairwise patient-clustered bootstrap matrix: heatmap (upper triangle,
    green/pale-red = CI excludes/includes 0) + a summary table sized to fit every
    pair -- no height-based cutoff, since that would silently drop whichever pairs
    sort to the bottom (see the DeLong-matrix table bug this mirrors the fix for)."""
    names = p_matrix_boot.index.tolist()  # RAW keys -- join key into bootstrap_matrix_df, do not map
    labels = config_labels(names)  # display only, same order as names
    n = len(names)
    n_pairs = len(bootstrap_matrix_df)
    max_name_len = max((len(s) for s in labels), default=8)
    fig_width = max(14, 9 + max_name_len * 0.35)
    fig_height = max(5, n * 0.5, 1.3 + n_pairs * 0.22)

    fig, (ax_heat, ax_table) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    diag_gray = mcolors.to_rgba("#e1e0d9")
    sig_color = mcolors.to_rgba(STATUS_SIGNIFICANT)
    not_sig_color = mcolors.to_rgba(STATUS_NOT_SIGNIFICANT)
    colors = np.ones((n, n, 4))
    for i in range(n):
        for j in range(n):
            if i >= j or pd.isna(p_matrix_boot.iloc[i, j]):
                colors[i, j] = diag_gray
            else:
                colors[i, j] = sig_color if p_matrix_boot.iloc[i, j] < 0.5 else not_sig_color
    ax_heat.imshow(colors, aspect="auto")
    for i in range(n):
        for j in range(i + 1, n):
            if pd.isna(p_matrix_boot.iloc[i, j]):
                continue
            row = bootstrap_matrix_df[
                (bootstrap_matrix_df["config_a"] == names[i])
                & (bootstrap_matrix_df["config_b"] == names[j])
            ].iloc[0]
            ax_heat.text(
                j,
                i,
                f"{row['delta_auroc']:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if row["significant"] else "normal",
                color="#0b0b0b",
            )
        ax_heat.text(i, i, "-", ha="center", va="center", fontsize=10, color=STATUS_NEUTRAL)
    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(labels, fontsize=9)
    ax_heat.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.5)
    ax_heat.tick_params(which="minor", length=0)
    ax_heat.set_title(
        f"Patient-clustered bootstrap DeltaAUROC (A - B), heldout ensemble, clip-level, "
        f"EXPLORATORY (95% CI, alpha={alpha})",
        fontsize=11,
    )
    legend_elems = [
        mpatches.Patch(facecolor=STATUS_SIGNIFICANT, label="CI excludes 0 (significant)"),
        mpatches.Patch(facecolor=STATUS_NOT_SIGNIFICANT, label="CI includes 0 (not significant)"),
    ]
    ax_heat.legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    ax_table.axis("off")
    y = 0.97
    ax_table.text(
        0.05,
        y,
        "Patient-clustered bootstrap, all pairs",
        fontsize=11,
        fontweight="bold",
        transform=ax_table.transAxes,
        va="top",
    )
    y -= 0.06
    headers = ["Model A", "Model B", "ΔAUROC", "95% CI", "sig."]
    name_col_width = min(0.30, 0.09 + max_name_len * 0.011)
    col_x = [
        0.00,
        name_col_width,
        2 * name_col_width,
        2 * name_col_width + 0.14,
        2 * name_col_width + 0.34,
    ]
    for hdr, x in zip(headers, col_x):
        ax_table.text(
            x + 0.02,
            y,
            hdr,
            color=STATUS_NEUTRAL,
            fontsize=8,
            fontweight="bold",
            transform=ax_table.transAxes,
            va="top",
        )
    y -= 0.04
    ax_table.axhline(y, color="#e1e0d9", linewidth=0.8, xmin=0.02, xmax=0.98)
    y -= 0.01

    row_step = y / max(n_pairs, 1)
    for _, row in bootstrap_matrix_df.iterrows():
        color = STATUS_SIGNIFICANT if row["significant"] else "#0b0b0b"
        vals = [
            config_label(row["config_a"]),
            config_label(row["config_b"]),
            f"{row['delta_auroc']:+.4f}",
            f"({row['ci_lower']:+.4f}, {row['ci_upper']:+.4f})",
            "Yes" if row["significant"] else "No",
        ]
        for val, x in zip(vals, col_x):
            ax_table.text(
                x + 0.02, y, val, color=color, fontsize=7.5, transform=ax_table.transAxes, va="top"
            )
        y -= row_step

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dispersion_table(df: pd.DataFrame, output_path) -> None:
    """Render the frame/clip fold-dispersion table (mean +/- std across the 5
    individually-trained fold models, not the ensemble) as TWO stacked
    tables in one figure -- frame above, clip below -- each with only its
    own 6 metric columns (Config + 6, not Config + 12 in one very wide row).

    Config is the long display name, which already states architecture/
    pretrain data/method, so those no longer need their own columns (they
    stay as their own columns in the underlying CSV/MD)."""

    def _level_table(level: str) -> tuple[list, list]:
        header = ["Config"] + [metric_label(m) for m in _METRIC_ORDER]
        cell_text = []
        for _, row in df.iterrows():
            cells = [config_label(row["config"], short=False)]
            for m in _METRIC_ORDER:
                cells.append(f"{row[f'{m}_{level}_mean']:.3f} +/- {row[f'{m}_{level}_std']:.3f}")
            cell_text.append(cells)
        return header, cell_text

    header, cells_frame = _level_table("frame")
    _, cells_clip = _level_table("clip")

    # ax.table does not size columns by content; derive explicit colWidths
    # and figure width from the actual longest string per column, shared by
    # both tables since they have identical headers -- same fix as
    # cv_results.py's main table (long Config strings clipped otherwise).
    col_max_len = [
        max(len(str(row[j])) for row in [header] + cells_frame + cells_clip)
        for j in range(len(header))
    ]
    col_widths = [n / sum(col_max_len) for n in col_max_len]
    fig_w = max(11.0, 1.0 + 0.145 * sum(col_max_len))
    fig_h = max(3.4, 0.9 + 0.45 * len(df) * 2)

    fig, (ax_frame, ax_clip) = plt.subplots(2, 1, figsize=(fig_w, fig_h))
    for ax, cell_text, level_title in (
        (ax_frame, cells_frame, "Frame level"),
        (ax_clip, cells_clip, "Clip level"),
    ):
        ax.axis("off")
        table = ax.table(
            cellText=cell_text,
            colLabels=header,
            colWidths=col_widths,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.6)
        for j in range(len(header)):
            table[0, j].set_facecolor("#e1e0d9")
            table[0, j].set_text_props(fontweight="bold")
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            table[i, 0].set_facecolor(config_color(row["config"]))
        ax.set_title(level_title, fontsize=10.5, fontweight="bold", pad=8)

    fig.suptitle(
        "Heldout test-set performance, mean +/- std across the 5 CV fold models",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest / clip structure
# ---------------------------------------------------------------------------


def _load_clip_structure(manifest_path) -> tuple:
    df_hm = pd.read_csv(manifest_path)
    clip_order = sorted(df_hm["clip_key"].unique().tolist())
    clip_label_map = df_hm.groupby("clip_key")["label"].apply(lambda s: int(s.mode()[0]))
    clip_patient_map = df_hm.groupby("clip_key")["patient_id"].first()
    clip_labels = np.array([clip_label_map[c] for c in clip_order], dtype=int)
    clip_patient_ids = np.array([clip_patient_map[c] for c in clip_order])
    clip_keys_frame = df_hm["clip_key"].to_numpy()
    manifest_frame_labels = df_hm["label"].to_numpy()
    # Per-FRAME patient id (one row per frame, same manifest row order as
    # manifest_frame_labels) -- enables a patient-clustered bootstrap on
    # frame-level ensemble metrics, resampling the same 19 patients as the
    # clip-level bootstrap does, just evaluating the metric over frames.
    manifest_frame_patient_ids = df_hm["patient_id"].to_numpy()
    return (
        clip_order,
        clip_labels,
        clip_patient_ids,
        clip_keys_frame,
        manifest_frame_labels,
        manifest_frame_patient_ids,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_experiment(
    experiment_name: str,
    manifest_path: str,
    output_dir,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    from pathlib import Path

    output_dir = Path(output_dir)
    bootstrap_dir = output_dir / "bootstrap"

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    for keys in ([p["config_a"], p["config_b"]] for p in PAIRS):
        for k in keys:
            get_model_entry(k)  # fail fast on a typo'd registry key

    print(f"Loading clip structure from {manifest_path} ...")
    (
        clip_order,
        clip_labels,
        clip_patient_ids,
        clip_keys_frame,
        manifest_frame_labels,
        manifest_frame_patient_ids,
    ) = _load_clip_structure(manifest_path)
    n_clips, n_patients = len(clip_order), len(np.unique(clip_patient_ids))
    print(f"  {n_patients} patients, {n_clips} clips, {len(clip_keys_frame)} frames.")

    print(f"\nLoading heldout predictions for {len(MODEL_REGISTRY)} configuration(s)...")
    configs = _load_all_configs(
        client, exp.experiment_id, clip_order, clip_keys_frame, clip_labels, manifest_frame_labels
    )
    if not configs:
        print("No configurations could be loaded, aborting.")
        return

    # ------------------------------------------------------------------
    # Per-fold table
    # ------------------------------------------------------------------
    per_fold_rows = []
    for model_key, data in configs.items():
        entry = get_model_entry(model_key)
        for i, (m, thr, m_frame) in enumerate(
            zip(data["fold_metrics"], data["fold_thresholds"], data["fold_metrics_frame"]), start=1
        ):
            per_fold_rows.append(
                {
                    "config": model_key,
                    "architecture": entry.architecture,
                    "pretrain_data": entry.pretrain_data,
                    "pretrain_method": entry.pretrain_method,
                    "fold": i,
                    "threshold": round(thr, 4),
                    **{k: round(v, 4) for k, v in m.items()},
                    **{f"{k}_frame": round(v, 4) for k, v in m_frame.items()},
                }
            )
    per_fold_df = pd.DataFrame(per_fold_rows)
    print(f"\nPer-fold table: {len(per_fold_df)} rows.")

    # ------------------------------------------------------------------
    # Frame- and clip-level heldout performance, mean +/- std across the 5
    # CV fold models (not the ensemble -- this is the individually-trained
    # fold models' dispersion, the descriptive counterpart to the ensemble
    # table below).
    # ------------------------------------------------------------------
    dispersion_rows = []
    for model_key, data in configs.items():
        entry = get_model_entry(model_key)
        row = {
            "config": model_key,
            "architecture": entry.architecture,
            "pretrain_data": entry.pretrain_data,
            "pretrain_method": entry.pretrain_method,
            "n_folds": len(data["fold_metrics"]),
        }
        for level, fold_metrics_list in (
            ("frame", data["fold_metrics_frame"]),
            ("clip", data["fold_metrics"]),
        ):
            for metric_name in _METRIC_ORDER:
                vals = np.array([fm[metric_name] for fm in fold_metrics_list], dtype=float)
                vals = vals[~np.isnan(vals)]
                row[f"{metric_name}_{level}_mean"] = (
                    round(float(vals.mean()), 4) if len(vals) else float("nan")
                )
                row[f"{metric_name}_{level}_std"] = (
                    round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
                )
        dispersion_rows.append(row)
    dispersion_df = pd.DataFrame(dispersion_rows)
    print(f"Frame/clip fold-dispersion table: {len(dispersion_df)} configs.")

    # ------------------------------------------------------------------
    # Per-config table (patient-clustered bootstrap on ALL 6 ensemble metrics,
    # at BOTH levels -- AUROC is threshold-free; the other 5 use the fixed
    # cv_mean_clip_threshold / cv_mean_threshold, the same threshold as their
    # point estimate, so the CI reflects patient-cohort sampling variability,
    # not threshold re-selection. Clip-level remains the PRIMARY grain (it is
    # what the pairwise effect-decomposition below tests); frame-level is a
    # descriptive companion that now also carries a CI, using the exact same
    # patient-clustered resampling -- resampling the 19 patients and
    # evaluating the metric over that patient subset's FRAMES each time, not
    # a naive per-frame bootstrap, so it does not pseudo-replicate frames.
    # ------------------------------------------------------------------
    print(f"\nBootstrapping ensemble metrics per config (n_bootstrap={n_bootstrap})...")
    config_bootstrap: dict = {}  # model_key -> {metric_name -> clip-level bootstrap result}
    config_bootstrap_frame: dict = {}  # model_key -> {metric_name -> frame-level bootstrap result}
    per_config_rows = []
    for model_key, data in configs.items():
        entry = get_model_entry(model_key)
        ens_probs = data["ensemble_clip_probs"]
        threshold = data["cv_mean_clip_threshold"]
        frame_threshold = data["cv_mean_threshold"]

        metric_results = {}
        frame_metric_results = {}
        for metric_name in _METRIC_ORDER:
            fn = _make_metric_bootstrap_fn(metric_name, clip_labels, ens_probs, threshold)
            metric_results[metric_name] = patient_clustered_bootstrap(
                clip_labels, clip_patient_ids, fn, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
            )
            fn_frame = _make_metric_bootstrap_fn(
                metric_name, manifest_frame_labels, data["ensemble_frame_probs"], frame_threshold
            )
            frame_metric_results[metric_name] = patient_clustered_bootstrap(
                manifest_frame_labels,
                manifest_frame_patient_ids,
                fn_frame,
                n_bootstrap=n_bootstrap,
                seed=seed,
                alpha=alpha,
            )
        config_bootstrap[model_key] = metric_results
        config_bootstrap_frame[model_key] = frame_metric_results

        best_fold_auroc = (
            float(roc_auc_score(clip_labels, data["best_fold_clip_probs"]))
            if len(np.unique(clip_labels)) > 1
            else float("nan")
        )

        row = {
            "config": model_key,
            "architecture": entry.architecture,
            "pretrain_data": entry.pretrain_data,
            "pretrain_method": entry.pretrain_method,
            "threshold": round(threshold, 4),
            "threshold_frame": round(frame_threshold, 4),
            "best_fold_auroc": round(best_fold_auroc, 4),
        }
        for metric_name in _METRIC_ORDER:
            r = metric_results[metric_name]
            row[metric_name] = round(r["point"], 4)
            row[f"{metric_name}_ci_percentile_lower"] = round(r["ci_percentile"][0], 4)
            row[f"{metric_name}_ci_percentile_upper"] = round(r["ci_percentile"][1], 4)
            row[f"{metric_name}_ci_bca_lower"] = round(r["ci_bca"][0], 4)
            row[f"{metric_name}_ci_bca_upper"] = round(r["ci_bca"][1], 4)
            row[f"{metric_name}_n_bootstrap_dropped"] = r["n_dropped"]

            rf = frame_metric_results[metric_name]
            row[f"{metric_name}_frame"] = round(rf["point"], 4)
            row[f"{metric_name}_frame_ci_percentile_lower"] = round(rf["ci_percentile"][0], 4)
            row[f"{metric_name}_frame_ci_percentile_upper"] = round(rf["ci_percentile"][1], 4)
            row[f"{metric_name}_frame_ci_bca_lower"] = round(rf["ci_bca"][0], 4)
            row[f"{metric_name}_frame_ci_bca_upper"] = round(rf["ci_bca"][1], 4)
            row[f"{metric_name}_frame_n_bootstrap_dropped"] = rf["n_dropped"]
        per_config_rows.append(row)

        a, af = metric_results["auroc"], frame_metric_results["auroc"]
        print(
            f"  {model_key:28s} thr={threshold:.3f}  AUROC={a['point']:.4f}  "
            f"pct=({a['ci_percentile'][0]:.4f}-{a['ci_percentile'][1]:.4f})  "
            f"BCa=({a['ci_bca'][0]:.4f}-{a['ci_bca'][1]:.4f})  "
            f"dropped={a['n_dropped']}/{n_bootstrap}  |  "
            f"frame thr={frame_threshold:.3f} AUROC={af['point']:.4f}  "
            f"pct=({af['ci_percentile'][0]:.4f}-{af['ci_percentile'][1]:.4f})"
        )
    per_config_df = pd.DataFrame(per_config_rows)

    # ------------------------------------------------------------------
    # Pairwise effect-decomposition table
    # ------------------------------------------------------------------
    print(f"\nComputing effect decomposition for {len(PAIRS)} pair(s)...")
    pair_bootstrap: dict = {}
    pair_rows = []
    ens_delong_p, best_delong_p = [], []
    for pair in PAIRS:
        a, b = pair["config_a"], pair["config_b"]
        if a not in configs or b not in configs:
            print(f"  [warn] {pair['pair_id']}: missing config(s), skipping.")
            continue

        ens_a, ens_b = configs[a]["ensemble_clip_probs"], configs[b]["ensemble_clip_probs"]
        bf_a, bf_b = configs[a]["best_fold_clip_probs"], configs[b]["best_fold_clip_probs"]

        def delta_fn(w, pa=ens_a, pb=ens_b):
            return float(
                roc_auc_score(clip_labels, pa, sample_weight=w)
                - roc_auc_score(clip_labels, pb, sample_weight=w)
            )

        boot = patient_clustered_bootstrap(
            clip_labels, clip_patient_ids, delta_fn, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
        )
        pair_bootstrap[pair["pair_id"]] = boot

        auc_a_ens, auc_b_ens, z_ens, p_ens = delong_test(clip_labels, ens_a, ens_b)
        auc_a_bf, auc_b_bf, z_bf, p_bf = delong_test(clip_labels, bf_a, bf_b)
        ens_delong_p.append(p_ens)
        best_delong_p.append(p_bf)

        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "effect": pair["effect"],
                "architecture": pair["architecture"],
                "config_a": a,
                "config_b": b,
                "auroc_a_ensemble": round(auc_a_ens, 4),
                "auroc_b_ensemble": round(auc_b_ens, 4),
                "delta_auroc_ensemble": round(boot["point"], 4),
                "boot_ci_percentile_lower": round(boot["ci_percentile"][0], 4),
                "boot_ci_percentile_upper": round(boot["ci_percentile"][1], 4),
                "boot_ci_bca_lower": round(boot["ci_bca"][0], 4),
                "boot_ci_bca_upper": round(boot["ci_bca"][1], 4),
                "n_bootstrap_dropped": boot["n_dropped"],
                "delong_ensemble_z": round(z_ens, 3),
                "delong_ensemble_p": round(p_ens, 4),
                "auroc_a_bestfold": round(auc_a_bf, 4),
                "auroc_b_bestfold": round(auc_b_bf, 4),
                "delta_auroc_bestfold": round(auc_a_bf - auc_b_bf, 4),
                "delong_bestfold_z": round(z_bf, 3),
                "delong_bestfold_p": round(p_bf, 4),
            }
        )
        print(
            f"  {pair['pair_id']:20s} DeltaAUROC={boot['point']:+.4f}  "
            f"pct=({boot['ci_percentile'][0]:+.4f},{boot['ci_percentile'][1]:+.4f})  "
            f"BCa=({boot['ci_bca'][0]:+.4f},{boot['ci_bca'][1]:+.4f})  "
            f"DeLong(ens) p={p_ens:.4f}  DeLong(best) p={p_bf:.4f}"
        )

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df["delong_ensemble_p_holm"] = np.round(holm_bonferroni(ens_delong_p), 4)
        pair_df["delong_bestfold_p_holm"] = np.round(holm_bonferroni(best_delong_p), 4)

    # ------------------------------------------------------------------
    # Full 9x9 patient-clustered bootstrap matrix, clip-level -- all 36
    # pairs among the 9 configs, exploratory/traceability only (same
    # caveat as everywhere else in this script: not corrected for multiple
    # comparisons, and not one of the 5 pre-specified architecture-matched
    # comparisons). Unlike a DeLong matrix this does NOT add a frame- or
    # clip-only independence violation on top of the all-pairs one -- it
    # reuses the same primary, patient-clustered methodology as the 5
    # pre-specified pairs, just applied to every pair instead of 5.
    # ------------------------------------------------------------------
    print(f"\nBootstrapping full 9x9 pairwise matrix, clip-level (n_bootstrap={n_bootstrap})...")
    matrix_rows = []
    config_keys = list(configs.keys())
    for a, b in itertools.combinations(config_keys, 2):
        probs_a, probs_b = configs[a]["ensemble_clip_probs"], configs[b]["ensemble_clip_probs"]

        def matrix_delta_fn(w, pa=probs_a, pb=probs_b):
            return float(
                roc_auc_score(clip_labels, pa, sample_weight=w)
                - roc_auc_score(clip_labels, pb, sample_weight=w)
            )

        boot = patient_clustered_bootstrap(
            clip_labels,
            clip_patient_ids,
            matrix_delta_fn,
            n_bootstrap=n_bootstrap,
            seed=seed,
            alpha=alpha,
        )
        lo, hi = boot["ci_percentile"]
        matrix_rows.append(
            {
                "config_a": a,
                "config_b": b,
                "delta_auroc": round(boot["point"], 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "significant": bool(lo > 0 or hi < 0),
            }
        )
    bootstrap_matrix_df = pd.DataFrame(matrix_rows).sort_values("ci_lower").reset_index(drop=True)
    p_matrix_boot = pd.DataFrame(np.nan, index=config_keys, columns=config_keys)
    for row in matrix_rows:
        p_matrix_boot.loc[row["config_a"], row["config_b"]] = 0.0 if row["significant"] else 1.0

    n_sig = int(bootstrap_matrix_df["significant"].sum())
    print(f"  {n_sig}/{len(bootstrap_matrix_df)} pairs significant (CI excludes 0)")

    # ------------------------------------------------------------------
    # Friedman + Wilcoxon on heldout AUROC (5-fold matrix), clip- and
    # frame-level -- clip is the primary grain used elsewhere in this
    # script; frame is reported alongside for the same reason the frame-
    # level metrics table exists (descriptive companion, not a second
    # significance claim -- clip-level pseudoreplicates less).
    #
    # Friedman only, no Wilcoxon here (unlike cv_results.py's validation-fold
    # side): the primary significance statements on the held-out cohort are
    # the patient-clustered bootstrap CIs above, and a 5-fold-paired Wilcoxon
    # on top of that would be a second, weaker, largely redundant test.
    # ------------------------------------------------------------------
    # (blocks x models) shape run_friedman/plot_friedman_ranks want: rows =
    # fold, columns = config. No Wilcoxon left in this file to also need the
    # (models x folds) shape, so build it this way directly instead of
    # transposing at each call site.
    print("\nRunning Friedman on heldout clip AUROC (per-fold matrix)...")
    fold_auroc_wide = per_fold_df.pivot(index="fold", columns="config", values="auroc")
    friedman_stat, friedman_p = run_friedman(fold_auroc_wide)
    print(f"  Friedman chi2={friedman_stat:.4f}  p={friedman_p:.4f}")

    print("Running Friedman on heldout frame AUROC (per-fold matrix)...")
    fold_auroc_wide_frame = per_fold_df.pivot(index="fold", columns="config", values="auroc_frame")
    friedman_stat_frame, friedman_p_frame = run_friedman(fold_auroc_wide_frame)
    print(f"  Friedman chi2={friedman_stat_frame:.4f}  p={friedman_p_frame:.4f}")

    # ------------------------------------------------------------------
    # Friedman at the PATIENT level: rows = the 19 heldout patients (blocks),
    # columns = the 9 model configs (treatments), values = each patient's mean
    # absolute error |true_label - predicted_probability| on the ensemble
    # CLIP predictions, averaged over that patient's clips. Unlike the
    # per-fold AUROC version above, mean absolute error is defined for every
    # patient regardless of class balance (2/19 patients have only one class
    # among their clips, so a per-patient AUROC would be undefined for them)
    # -- so all 19 patients contribute, giving this test its statistical
    # power. Lower error is better, so ranks are reversed (higher_is_better=False).
    # ------------------------------------------------------------------
    print("Running Friedman on heldout clip mean absolute error (per-patient matrix)...")
    patient_error_rows = []
    for patient in np.unique(clip_patient_ids):
        mask = clip_patient_ids == patient
        row = {"patient_id": patient}
        for model_key, data in configs.items():
            row[model_key] = float(
                np.mean(np.abs(clip_labels[mask] - data["ensemble_clip_probs"][mask]))
            )
        patient_error_rows.append(row)
    patient_error_df = pd.DataFrame(patient_error_rows).set_index("patient_id")
    friedman_stat_patient, friedman_p_patient = run_friedman(patient_error_df)
    print(
        f"  {len(patient_error_df)} patients  Friedman chi2={friedman_stat_patient:.4f}  "
        f"p={friedman_p_patient:.4f}"
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if dry_run:
        print("\n[DRY RUN] Skipping file writes.")
        print("\n--- Frame/clip fold-dispersion table ---")
        print(dispersion_df.to_string(index=False))
        print("\n--- Per-config table ---")
        print(per_config_df.to_string(index=False))
        print("\n--- Effect decomposition ---")
        print(pair_df.to_string(index=False))
        print("\n--- Full 9x9 bootstrap matrix (exploratory) ---")
        print(bootstrap_matrix_df.to_string(index=False))
        print("\n--- Per-patient mean absolute error ---")
        print(patient_error_df.to_string())
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    per_fold_df.to_csv(output_dir / "per_fold_heldout_metrics.csv", index=False)
    (output_dir / "per_fold_heldout_metrics.md").write_text(to_markdown(per_fold_df))

    dispersion_df.to_csv(output_dir / "heldout_fold_dispersion.csv", index=False)
    (output_dir / "heldout_fold_dispersion.md").write_text(to_markdown(dispersion_df))
    _plot_dispersion_table(dispersion_df, output_dir / "heldout_fold_dispersion.png")

    per_config_df.to_csv(output_dir / "heldout_ensemble_metrics.csv", index=False)
    (output_dir / "heldout_ensemble_metrics.md").write_text(to_markdown(per_config_df))

    pair_df.to_csv(output_dir / "effect_decomposition.csv", index=False)
    (output_dir / "effect_decomposition.md").write_text(to_markdown(pair_df))

    bootstrap_matrix_df.to_csv(output_dir / "bootstrap_matrix_heldout_clip.csv", index=False)
    _plot_bootstrap_matrix(
        p_matrix_boot, bootstrap_matrix_df, alpha, output_dir / "bootstrap_matrix_heldout_clip.png"
    )

    for model_key, metric_results in config_bootstrap.items():
        for metric_name, result in metric_results.items():
            np.save(bootstrap_dir / f"{model_key}_{metric_name}.npy", result["replicates"])
    for model_key, metric_results in config_bootstrap_frame.items():
        for metric_name, result in metric_results.items():
            np.save(bootstrap_dir / f"{model_key}_{metric_name}_frame.npy", result["replicates"])
    for pair_id, result in pair_bootstrap.items():
        np.save(bootstrap_dir / f"{pair_id}.npy", result["replicates"])

    names = per_config_df["config"].tolist()
    points = per_config_df["auroc"].tolist()
    cis_pct_auroc = list(
        zip(per_config_df["auroc_ci_percentile_lower"], per_config_df["auroc_ci_percentile_upper"])
    )
    cis_bca_auroc = list(
        zip(per_config_df["auroc_ci_bca_lower"], per_config_df["auroc_ci_bca_upper"])
    )
    _plot_bar_with_ci(
        names,
        points,
        cis_pct_auroc,
        "Heldout ensemble AUROC, clip-level (95% percentile CI)",
        "AUROC",
        output_dir / "heldout_ensemble_auroc_percentile.png",
    )
    _plot_bar_with_ci(
        names,
        points,
        cis_bca_auroc,
        "Heldout ensemble AUROC, clip-level (95% BCa CI)",
        "AUROC",
        output_dir / "heldout_ensemble_auroc_bca.png",
    )
    all_metrics_ci_pct = {
        m: list(
            zip(
                per_config_df[f"{m}_ci_percentile_lower"], per_config_df[f"{m}_ci_percentile_upper"]
            )
        )
        for m in _METRIC_ORDER
    }
    all_metrics_ci_pct_frame = {
        m: list(
            zip(
                per_config_df[f"{m}_frame_ci_percentile_lower"],
                per_config_df[f"{m}_frame_ci_percentile_upper"],
            )
        )
        for m in _METRIC_ORDER
    }
    _plot_two_level_grouped_metrics(
        names,
        {m: per_config_df[f"{m}_frame"].tolist() for m in _METRIC_ORDER},
        all_metrics_ci_pct_frame,
        {m: per_config_df[m].tolist() for m in _METRIC_ORDER},
        all_metrics_ci_pct,
        "Heldout ensemble metrics per configuration (95% percentile CI, patient-clustered)",
        output_dir / "heldout_ensemble_metrics_bar.png",
    )

    if not pair_df.empty:
        _plot_forest(
            pair_df,
            "boot_ci_percentile_lower",
            "boot_ci_percentile_upper",
            "Effect decomposition, clip-level (95% percentile CI)",
            output_dir / "effect_decomposition_percentile.png",
        )
        _plot_forest(
            pair_df,
            "boot_ci_bca_lower",
            "boot_ci_bca_upper",
            "Effect decomposition, clip-level (95% BCa CI)",
            output_dir / "effect_decomposition_bca.png",
        )

    # One multi-metric panel per architecture (all variants of that architecture side by side,
    # frame level on top / clip level on the bottom, both with 95% percentile CI --
    # same two-level treatment as the global heldout_ensemble_metrics_bar.png).
    for architecture, sub in per_config_df.groupby("architecture", sort=False):
        sub_ci_pct = {
            m: list(zip(sub[f"{m}_ci_percentile_lower"], sub[f"{m}_ci_percentile_upper"]))
            for m in _METRIC_ORDER
        }
        sub_ci_pct_frame = {
            m: list(
                zip(sub[f"{m}_frame_ci_percentile_lower"], sub[f"{m}_frame_ci_percentile_upper"])
            )
            for m in _METRIC_ORDER
        }
        _plot_two_level_grouped_metrics(
            sub["config"].tolist(),
            {m: sub[f"{m}_frame"].tolist() for m in _METRIC_ORDER},
            sub_ci_pct_frame,
            {m: sub[m].tolist() for m in _METRIC_ORDER},
            sub_ci_pct,
            f"Heldout ensemble metrics, {architecture} (95% percentile CI, patient-clustered)",
            output_dir / f"arch_{slug(architecture)}_heldout_metrics.png",
        )

    # Frame level (left) and clip level (right) side by side -- clip-level is
    # the unit statistical testing is done on; frame-level is a smoother
    # curve for presentation. The manuscript's significance claims must still
    # come from the clip-level / patient-clustered results above, not this.
    roc_data_clip = []
    for model_key, data in configs.items():
        probs = data["ensemble_clip_probs"]
        fpr, tpr, _ = roc_curve(clip_labels, probs)
        roc_data_clip.append(
            {
                "name": config_label(model_key),
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc_score(clip_labels, probs),
                "color": config_color(model_key),
            }
        )
    roc_data_frame = []
    for model_key, data in configs.items():
        probs = data["ensemble_frame_probs"]
        fpr, tpr, _ = roc_curve(manifest_frame_labels, probs)
        roc_data_frame.append(
            {
                "name": config_label(model_key),
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc_score(manifest_frame_labels, probs),
                "color": config_color(model_key),
            }
        )
    fig, (ax_frame, ax_clip) = plt.subplots(1, 2, figsize=(15.5, 8.2))
    plot_roc_curves(roc_data_frame, title="Frame level", ax=ax_frame)
    plot_roc_curves(roc_data_clip, title="Clip level (mean-probability aggregation)", ax=ax_clip)
    fig.suptitle("Heldout ensemble ROC curves", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(output_dir / "roc_curves_ensemble.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_friedman_ranks(
        fold_auroc_wide,
        friedman_stat,
        friedman_p,
        output_dir / "friedman_ranks_heldout.png",
        title="Model rankings, heldout clip AUROC (5-fold)",
    )
    plot_friedman_ranks(
        fold_auroc_wide_frame,
        friedman_stat_frame,
        friedman_p_frame,
        output_dir / "friedman_ranks_heldout_frame.png",
        title="Model rankings, heldout frame AUROC (5-fold)",
    )

    patient_error_df.to_csv(output_dir / "heldout_patient_mae.csv")
    plot_friedman_ranks(
        patient_error_df,
        friedman_stat_patient,
        friedman_p_patient,
        output_dir / "friedman_ranks_heldout_patient.png",
        title=f"Model rankings, heldout clip mean absolute error ({len(patient_error_df)} patients)",
        higher_is_better=False,
    )

    print(f"\nOutputs written to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="ulcer_detection")
    p.add_argument("--manifest", default=str(cfg.ulcer.heldout / "heldout_temporal_manifest.csv"))
    p.add_argument("--mlflow-uri", default=cfg.mlflow_db)
    p.add_argument("--output-dir", default=str(cfg.results_heldout_dir))
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Compute and print, skip writing files")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    if args.dry_run:
        print("[DRY RUN]")
    process_experiment(
        args.experiment,
        args.manifest,
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
        dry_run=args.dry_run,
    )
