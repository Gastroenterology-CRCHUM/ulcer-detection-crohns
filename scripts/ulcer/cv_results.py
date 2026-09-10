"""
scripts/ulcer/cv_results.py
----------------------------
Cross-validation (training/validation-fold) results across the 9 model
configurations (no re-inference).

Usage
-----
    python scripts/ulcer/cv_results.py
    python scripts/ulcer/cv_results.py --dry-run
    python scripts/ulcer/cv_results.py --no-val-specificity
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.config.models import MODEL_REGISTRY, get_model_entry
from src.config.paths import get_default_paths
from src.evaluation.metrics import METRIC_FNS
from src.evaluation.rank_tests import (
    plot_friedman_ranks,
    plot_wilcoxon_pmatrix,
    run_friedman,
    run_wilcoxon_matrix,
)
from src.evaluation.reporting import to_markdown
from src.evaluation.style import (
    STATUS_NEUTRAL,
    config_color,
    config_label,
    config_labels,
    metric_label,
    slug,
)

plt.style.use("seaborn-v0_8-whitegrid")

# Fallback when a fold logged no tuned threshold
CV_THRESHOLD = 0.5

# Parent-logged CV aggregates
_VAL_METRICS = ("auroc", "f1", "precision", "recall")

# Held-out per-fold metrics
_REPORT_METRICS = ("auroc", "sensitivity", "specificity", "f1")

# Per-fold held-out metrics are the only place in the CV runs where all four
# reporting metrics exist at BOTH frame and clip level: the validation folds log
# neither specificity nor any clip-level metric (see run_modes.py's CV loop,
# which logs only fold_val_{f1,precision,recall,auroc}), and validation
# clip-level cannot be recovered post-hoc because only val_probs/val_labels are
# saved per fold -- never the clip_ids needed to aggregate by clip.
_HELDOUT_LEVELS = {
    "frame": "heldout__{metric}_mean",
    "clip": "heldout_clip__{metric}_mean",
}


# ---------------------------------------------------------------------------
# MLflow loading -- metrics/params only, no artifacts
# ---------------------------------------------------------------------------


def _load_configs(client: MlflowClient, exp_id: str) -> dict:
    """Return {model_key: {"parent": Run, "children": [Run, ...]}} for CV parent runs."""
    all_runs = client.search_runs(experiment_ids=[exp_id], max_results=1000)
    parents_by_model = {}
    for r in all_runs:
        if r.data.tags.get("mlflow.parentRunId"):
            continue
        if not r.data.tags.get("training_mode", "").startswith("cv_"):
            continue
        parents_by_model[r.data.tags.get("model")] = r
    children_by_parent: dict = {}
    for r in all_runs:
        pid = r.data.tags.get("mlflow.parentRunId")
        if pid:
            children_by_parent.setdefault(pid, []).append(r)

    configs: dict = {}
    for model_key in MODEL_REGISTRY:
        parent = parents_by_model.get(model_key)
        if parent is None:
            print(f"  [warn] {model_key}: no CV parent run found, skipping.")
            continue
        children = sorted(
            children_by_parent.get(parent.info.run_id, []),
            key=lambda r: int(r.data.params.get("fold", -1)),
        )
        if not children:
            print(f"  [warn] {model_key}: no child fold runs, skipping.")
            continue
        configs[model_key] = {"parent": parent, "children": children}
    return configs


def _parent_metric(parent, key: str, default: float = float("nan")) -> float:
    return float(parent.data.metrics.get(key, default))


def _parent_param(parent, key: str, default=None):
    return parent.data.params.get(key, default)


# ---------------------------------------------------------------------------
# Validation-fold specificity/sensitivity, recomputed from artifacts
# ---------------------------------------------------------------------------

# Metrics recomputed from the val-fold probability artifacts. AUROC and F1 are
# recomputed too (not needed -- they are already logged) purely so each fold
# can be cross-checked against its logged values; a mismatch means the
# artifacts and the logged metrics disagree and the fold must not be trusted.
_VAL_ARTIFACT_METRICS = ("AUROC", "Sensitivity", "Specificity", "F1")

# Recomputed-vs-logged agreement tolerance. Logged metrics are full float64,
# so exact agreement is expected; 1e-4 only absorbs float round-trips.
_CROSSCHECK_TOL = 1e-4


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

    # Cross-check the three metrics run_modes.py already logged.
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


# ---------------------------------------------------------------------------
# Plotting -- shared bar-chart primitives
# ---------------------------------------------------------------------------


def _draw_grouped_bars(ax, keys, labels, metric_values: dict, metric_stds: dict, xlabels) -> None:
    """Draw one grouped-bar panel: metrics on the x-axis, one bar per model,
    error bars = std across folds. Used by the held-out two-level chart.

    keys/labels are separate parameters, never one list: config_color raises
    on anything that isn't a raw MODEL_REGISTRY key, so the color lookup
    always uses `keys` while the legend text uses `labels`.

    Values/stds may be NaN (a config whose folds never logged that metric);
    matplotlib skips NaN bars, but a NaN in ``yerr`` blows up the errorbar
    transform, so stds are zeroed out for drawing only.
    """
    metric_names = list(metric_values.keys())
    n_models = len(keys)
    x = np.arange(len(metric_names))
    width = 0.8 / n_models
    for i, (key, label) in enumerate(zip(keys, labels)):
        values = [metric_values[m][i] for m in metric_names]
        yerr = np.nan_to_num(
            np.array([metric_stds[m][i] for m in metric_names], dtype=float), nan=0.0
        )
        ax.bar(
            x + i * width,
            values,
            width,
            yerr=yerr,
            capsize=2.5,
            label=label,
            color=config_color(key),
            # Thin mid-gray whiskers: at 9 configs x 4 metrics the default
            # heavy black errorbars visually outweigh the bars they annotate.
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#4a4a4a"},
        )
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=0)
    # Bars stay zero-baselined -- truncating the axis to the 0.7-1.0 band where
    # the values live would exaggerate the between-config differences.
    ax.set_ylim(0, 1.05)


def _plot_model_bar(
    keys, means, stds, ylabel, title, output_path, *, ref_line=None, ref_label=None
) -> None:
    """One bar per model, mean +/- std, colored by config_color, config_label
    rotated on the x-axis. Used for threshold dispersion, where the values
    (a hyperparameter, not a performance claim) are worth spreading out."""
    fig, ax = plt.subplots(figsize=(max(9, len(keys) * 1.0), 5.5))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=4, color=[config_color(k) for k in keys])
    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="#c3c2b7", linewidth=1, label=ref_label)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels(keys), rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_auroc_legend_bar(keys, means, stds, title, output_path, *, ref_line=None, ref_label=None) -> None:
    """One tightly-packed cluster of bars (all models in a single 'AUROC'
    x-slot, narrow width) -- model identity AND its numeric mean +/- std move
    into the legend instead of rotated x-tick labels, since with only one
    metric there is nothing else competing for x-axis space."""
    labels = [f"{config_label(k)}  ({m:.3f} +/- {s:.3f})" for k, m, s in zip(keys, means, stds)]
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    _draw_grouped_bars(
        ax, keys, labels, {"auroc": list(means)}, {"auroc": list(stds)}, [metric_label("auroc")]
    )
    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="#c3c2b7", linewidth=1, label=ref_label)
    ax.set_ylabel("Validation-fold AUROC (mean +/- std across folds)")
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0.30, 1, 1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_two_level_grouped_bar(keys, stats: dict, title, output_path) -> None:
    """AUROC / Sensitivity / Specificity / F1 as mean +/- std across folds, with
    frame level on the top panel and clip level on the bottom panel.

    Args:
        keys:   Config keys, one bar per config in each metric group.
        stats:  {level: (metric_values, metric_stds)} for levels "frame" and
                "clip", each a {metric: [value per config]} dict.
        title:  Figure suptitle.
    """
    labels = config_labels(keys)
    xlabels = [metric_label(m) for m in _REPORT_METRICS]
    fig, axes = plt.subplots(
        2, 1, figsize=(max(11, len(_REPORT_METRICS) * 2.8), 8.6), sharex=True, sharey=True
    )
    panel_titles = {"frame": "Frame level", "clip": "Clip level (mean-probability aggregation)"}

    for ax, level in zip(axes, ("frame", "clip")):
        values, stds = stats[level]
        _draw_grouped_bars(ax, keys, labels, values, stds, xlabels)
        ax.set_title(panel_titles[level], fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (mean +/- std)", fontsize=9)
        # A config can be missing a whole level (clip metrics were backfilled
        # post-hoc and cover fewer folds); say so rather than showing a gap
        # that reads like a score of zero.
        if all(np.isnan(values[m]).all() for m in _REPORT_METRICS):
            ax.text(
                0.5,
                0.5,
                f"No {level}-level metrics logged for these configurations",
                ha="center",
                va="center",
                fontsize=11,
                color=STATUS_NEUTRAL,
                transform=ax.transAxes,
            )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8, frameon=True
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting -- CV fold x model AUROC matrix
# ---------------------------------------------------------------------------

def _plot_fold_matrix(df: pd.DataFrame, output_path) -> None:
    """Fold x model AUROC as a plain table, row order = architecture (as
    given -- callers pass a frame already reindexed to MODEL_REGISTRY order,
    matching every other table in this file). Each Config cell keeps its
    config_color facecolor; the maximum value in each Fold column is bolded
    so the reader can spot which model won each fold at a glance. A "Fold
    mean" row is appended in italic neutral text -- fold difficulty and
    model quality are otherwise superimposed with no way to separate them by
    eye (a fold that is uniformly easier/harder for every model can look
    like a model effect)."""
    fold_cols = [c for c in df.columns if c not in ("Mean", "Std")]
    header = ["Config"] + [f"Fold {c.split('_')[-1]}" for c in fold_cols] + ["Mean", "Std"]

    fold_values = df[fold_cols].to_numpy(dtype=float)
    col_max = np.nanmax(fold_values, axis=0)  # per-fold-column max, for bolding

    cell_text, bold_mask = [], []
    for config_key, row in df.iterrows():
        cells = [config_label(config_key, short=False)]
        bmask = [False]
        for j, c in enumerate(fold_cols):
            v = row[c]
            cells.append("n/a" if pd.isna(v) else f"{v:.3f}")
            bmask.append(bool(not pd.isna(v) and np.isclose(v, col_max[j])))
        cells.append(f"{row['Mean']:.3f}")
        bmask.append(False)
        cells.append(f"{row['Std']:.3f}" if not pd.isna(row["Std"]) else "--")
        bmask.append(False)
        cell_text.append(cells)
        bold_mask.append(bmask)

    fold_means = np.nanmean(fold_values, axis=0)
    cell_text.append(["Fold mean"] + [f"{v:.3f}" for v in fold_means] + ["", ""])
    bold_mask.append([False] * len(header))

    # ax.table does not size columns by content; derive explicit colWidths
    # and figure width from the actual longest string per column (same fix
    # as _plot_main_table -- the long Config strings clip otherwise).
    col_max_len = [
        max(len(str(row[j])) for row in [header] + cell_text) for j in range(len(header))
    ]
    col_widths = [n / sum(col_max_len) for n in col_max_len]
    fig_w = max(9.0, 1.2 + 0.145 * sum(col_max_len))
    fig_h = max(3.2, 0.55 + 0.42 * len(cell_text))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, colLabels=header, colWidths=col_widths, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    for j in range(len(header)):
        table[0, j].set_facecolor("#e1e0d9")
        table[0, j].set_text_props(fontweight="bold")
    for i, config_key in enumerate(df.index, start=1):
        table[i, 0].set_facecolor(config_color(config_key))
    for i, bmask in enumerate(bold_mask, start=1):
        for j, is_bold in enumerate(bmask):
            if is_bold:
                table[i, j].set_text_props(fontweight="bold")
    footer_row = len(cell_text)
    for j in range(len(header)):
        table[footer_row, j].set_text_props(style="italic", color=STATUS_NEUTRAL)

    ax.set_title("CV validation-fold AUROC per model, frame-level", fontsize=11, pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting -- main table
# ---------------------------------------------------------------------------


def _plot_main_table(df: pd.DataFrame, output_path) -> None:
    """Render the main CV comparison table as an image (not just CSV/MD) so
    there's a single figure a reader can drop straight into a slide/manuscript.

    AUROC only (mean +/- std): the one threshold-free validation-fold metric,
    see module docstring. Sensitivity/specificity/F1/precision stay out of
    this headline table -- they are threshold-dependent and that threshold
    was tuned on this same validation fold, which is not a fair cross-model
    comparison. They remain in cv_per_fold_val_metrics.csv as diagnostic
    detail. The Config column is the long display name, which already states
    architecture/pretrain data/method, so those no longer need separate
    columns here (they stay as their own columns in the CSV/MD).
    """
    header = ["Config", metric_label("auroc"), "Best fold", "Best-fold AUROC", "Best-fold thr"]

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append(
            [
                config_label(row["config"], short=False),
                f"{row['val_auroc_mean']:.3f} +/- {row['val_auroc_std']:.3f}",
                str(row["best_fold"]),
                f"{row['best_fold_val_auroc']:.3f}",
                f"{row['best_fold_threshold']:.3f}",
            ]
        )

    # ax.table does NOT size columns by content -- it splits the axes evenly
    # by default, which clips the long Config strings (up to 54 chars for
    # "ViT-Small/16, ImageNet-1K / Supervised (timm AugReg)"). Derive explicit
    # colWidths and the figure width from the actual longest string per
    # column instead of a fixed per-column inch guess.
    col_max_len = [
        max(len(str(row[j])) for row in [header] + cell_text) for j in range(len(header))
    ]
    col_widths = [n / sum(col_max_len) for n in col_max_len]
    fig_w = max(11.0, 1.6 + 0.11 * sum(col_max_len))

    fig, ax = plt.subplots(figsize=(fig_w, 0.5 + 0.4 * len(df)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, colLabels=header, colWidths=col_widths, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    for j in range(len(header)):
        table[0, j].set_facecolor("#e1e0d9")
        table[0, j].set_text_props(fontweight="bold")
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        table[i, 0].set_facecolor(config_color(row["config"]))
    ax.set_title(
        "CV main table: validation-fold AUROC (mean +/- std) and best fold",
        fontsize=11,
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _summarise_heldout_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fold held-out metrics into mean/std across folds per config.

    Returns one row per config with heldout_{level}_{metric}_{mean,std} columns
    plus n_folds_{level}, the number of folds that actually logged that level
    (clip metrics were backfilled post-hoc and cover fewer folds than frame).
    Std uses ddof=1, matching the CV aggregates logged by run_modes.py, so a
    single-fold config yields NaN rather than a misleading 0.
    """
    rows = []
    for (model_key, architecture), sub in fold_df.groupby(
        ["config", "architecture"], sort=False
    ):
        row: dict = {"config": model_key, "architecture": architecture}
        for level in _HELDOUT_LEVELS:
            cols = [f"heldout_{level}_{m}" for m in _REPORT_METRICS]
            row[f"n_folds_{level}"] = int(sub[cols].notna().any(axis=1).sum())
            for m, col in zip(_REPORT_METRICS, cols):
                values = sub[col].dropna()
                row[f"heldout_{level}_{m}_mean"] = (
                    round(float(values.mean()), 4) if len(values) else float("nan")
                )
                row[f"heldout_{level}_{m}_std"] = (
                    round(float(values.std(ddof=1)), 4) if len(values) > 1 else float("nan")
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _heldout_panel_stats(df: pd.DataFrame) -> dict:
    """Reshape a held-out summary frame into the {level: (values, stds)} form
    _plot_two_level_grouped_bar expects."""
    return {
        level: (
            {m: df[f"heldout_{level}_{m}_mean"].to_numpy(dtype=float) for m in _REPORT_METRICS},
            {m: df[f"heldout_{level}_{m}_std"].to_numpy(dtype=float) for m in _REPORT_METRICS},
        )
        for level in _HELDOUT_LEVELS
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_experiment(
    experiment_name: str,
    output_dir,
    alpha: float = 0.05,
    dry_run: bool = False,
    with_val_specificity: bool = True,
    heldout_output_dir=None,
) -> None:
    cfg = get_default_paths()
    output_dir = Path(output_dir)
    # Held-out figures belong with the other held-out results, not the CV ones.
    heldout_dir = Path(heldout_output_dir) if heldout_output_dir else cfg.results_heldout_dir
    artifact_misses: list[str] = []
    crosscheck_failures: list[str] = []

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    print(f"Loading CV runs for {len(MODEL_REGISTRY)} configuration(s)...")
    configs = _load_configs(client, exp.experiment_id)
    if not configs:
        print("No CV configurations could be loaded, aborting.")
        return

    # ------------------------------------------------------------------
    # Main table (per config, parent-logged CV aggregates)
    # ------------------------------------------------------------------
    main_rows = []
    fold_rows = []
    for model_key, data in configs.items():
        entry = get_model_entry(model_key)
        parent = data["parent"]
        row = {
            "config": model_key,
            "architecture": entry.architecture,
            "pretrain_data": entry.pretrain_data,
            "pretrain_method": entry.pretrain_method,
            "n_folds": len(data["children"]),
        }
        for m in _VAL_METRICS:
            row[f"val_{m}_mean"] = round(_parent_metric(parent, f"cv_mean_val_{m}"), 4)
            row[f"val_{m}_std"] = round(_parent_metric(parent, f"cv_std_val_{m}"), 4)
        row["threshold_mean"] = round(_parent_metric(parent, "cv_mean_threshold"), 4)
        row["threshold_std"] = round(_parent_metric(parent, "cv_std_threshold"), 4)
        row["threshold_min"] = round(_parent_metric(parent, "cv_min_threshold"), 4)
        row["threshold_max"] = round(_parent_metric(parent, "cv_max_threshold"), 4)
        # run_modes.py logs the "best_fold" PARAM 0-indexed (it prints best_fold+1
        # but logs the raw index, see run_modes.py:890-895). Every other fold
        # reference in this script is 1-based -- including the per-fold table's
        # "fold" column below -- so normalise to 1-based here.
        best_fold_raw = _parent_param(parent, "best_fold")
        row["best_fold"] = int(best_fold_raw) + 1 if best_fold_raw is not None else None
        best_fold_auroc = _parent_param(parent, "best_fold_val_auroc")
        row["best_fold_val_auroc"] = (
            round(float(best_fold_auroc), 4) if best_fold_auroc else float("nan")
        )
        best_fold_thr = _parent_param(parent, "best_fold_threshold")
        row["best_fold_threshold"] = (
            round(float(best_fold_thr), 4) if best_fold_thr else float("nan")
        )
        main_rows.append(row)

        for child in data["children"]:
            fold_idx = int(child.data.params.get("fold", -1))
            fold_row = {
                "config": model_key,
                "architecture": entry.architecture,
                "fold": fold_idx + 1,
                "val_auroc": child.data.metrics.get("fold_val_auroc"),
                "val_f1": child.data.metrics.get("fold_val_f1"),
                "val_precision": child.data.metrics.get("fold_val_precision"),
                "val_recall": child.data.metrics.get("fold_val_recall"),
                "optimal_threshold": child.data.metrics.get("fold_optimal_threshold"),
            }
            # Sensitivity is identical to the already-logged recall; specificity
            # is not logged at all and has to come from the probability
            # artifacts. Fall back to the logged recall for sensitivity so a
            # missing artifact costs only specificity, not both.
            fold_row["val_sensitivity"] = child.data.metrics.get("fold_val_recall")
            fold_row["val_specificity"] = None
            if with_val_specificity:
                recomputed = _val_fold_metrics_from_artifacts(child, fold_idx)
                if recomputed is None:
                    artifact_misses.append(f"{model_key} fold {fold_idx + 1}")
                else:
                    fold_row["val_sensitivity"] = recomputed["sensitivity"]
                    fold_row["val_specificity"] = recomputed["specificity"]
                    for msg in recomputed["_mismatch"]:
                        crosscheck_failures.append(f"{model_key} fold {fold_idx + 1}: {msg}")
            for level, key_template in _HELDOUT_LEVELS.items():
                for m in _REPORT_METRICS:
                    fold_row[f"heldout_{level}_{m}"] = child.data.metrics.get(
                        key_template.format(metric=m)
                    )
            fold_rows.append(fold_row)

    main_df = pd.DataFrame(main_rows)
    fold_df = pd.DataFrame(fold_rows)

    if with_val_specificity:
        if artifact_misses:
            print(
                f"\n  [warn] {len(artifact_misses)} fold(s) had no readable val "
                f"probability artifact, excluded from val specificity: "
                f"{', '.join(artifact_misses)}"
            )
        if crosscheck_failures:
            print(
                f"\n  [ERROR] {len(crosscheck_failures)} fold(s) where recomputed "
                f"metrics disagree with the logged ones -- their specificity is "
                f"NOT trustworthy:"
            )
            for msg in crosscheck_failures:
                print(f"    - {msg}")
        else:
            n_ok = int(fold_df["val_specificity"].notna().sum())
            print(
                f"\n  Val specificity recomputed for {n_ok}/{len(fold_df)} folds "
                f"(cv_per_fold_val_metrics.csv); recomputed sensitivity/F1/AUROC "
                f"match the logged values on all of them."
            )

    print(f"\nMain table: {len(main_df)} configs, per-fold table: {len(fold_df)} rows.")
    print(main_df.to_string(index=False))

    heldout_df = _summarise_heldout_folds(fold_df)
    print(f"\nHeld-out per-fold summary ({len(heldout_df)} configs):")
    print(heldout_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Friedman + Wilcoxon on validation-fold AUROC (the fair, threshold-free
    # metric -- see module docstring). Supersedes the old
    # scripts/ulcer/statistical_comparison.py, which read the metric off the
    # wrong run (parent, where it was never logged) and depended on a
    # scikit_posthocs/statsmodels combination broken in this environment.
    # ------------------------------------------------------------------
    fold_auroc_wide = fold_df.pivot(index="config", columns="fold", values="val_auroc")
    # run_friedman/plot_friedman_ranks want (blocks x models) -- fold_auroc_wide
    # is (models x folds), the shape run_wilcoxon_matrix below needs, so transpose
    # only for the Friedman call.
    friedman_stat, friedman_p = run_friedman(fold_auroc_wide.T)
    wilcoxon_p = run_wilcoxon_matrix(fold_auroc_wide)
    print(f"\nFriedman (CV val AUROC): chi2={friedman_stat:.4f}  p={friedman_p:.4f}")

    if dry_run:
        print("\n[DRY RUN] Skipping file writes.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    main_df.to_csv(output_dir / "cv_main_table.csv", index=False)
    (output_dir / "cv_main_table.md").write_text(to_markdown(main_df))
    fold_df.to_csv(output_dir / "cv_per_fold_val_metrics.csv", index=False)
    wilcoxon_p.to_csv(output_dir / "wilcoxon_pmatrix_cv.csv")

    keys = main_df["config"].tolist()
    _plot_main_table(main_df, output_dir / "cv_main_table.png")
    _plot_auroc_legend_bar(
        keys,
        main_df["val_auroc_mean"].tolist(),
        main_df["val_auroc_std"].tolist(),
        "CV validation-fold AUROC per configuration, frame-level",
        output_dir / "cv_bar_chart.png",
        ref_line=0.5,
        ref_label="0.5 (chance)",
    )
    _plot_model_bar(
        keys,
        main_df["threshold_mean"].tolist(),
        main_df["threshold_std"].tolist(),
        "Optimal frame threshold (mean +/- std across folds)",
        "CV threshold dispersion across folds, frame-level",
        output_dir / "cv_threshold_dispersion.png",
        ref_line=0.5,
        ref_label="0.5 (default)",
    )

    # ------------------------------------------------------------------
    # Held-out per-fold metrics: AUROC / sensitivity / specificity / F1,
    # frame level above, clip level below. Global plus one per architecture.
    # All four metrics are fair here (fixed threshold, applied -- not tuned
    # -- on the held-out cohort), unlike the CV validation side above.
    #
    # These go to the held-out results directory, not the CV one -- they
    # describe the held-out cohort, not the validation folds. The "perfold"
    # in every filename distinguishes them from effect_decomposition.py's
    # ENSEMBLE figures in the same directory (arch_<slug>_heldout_metrics.png),
    # which average the 5 folds into one prediction and report bootstrap CIs
    # rather than a mean +/- std across folds. Different statistic, different
    # name, same directory.
    # ------------------------------------------------------------------
    heldout_dir.mkdir(parents=True, exist_ok=True)
    heldout_df.to_csv(heldout_dir / "heldout_perfold_summary.csv", index=False)
    _plot_two_level_grouped_bar(
        heldout_df["config"].tolist(),
        _heldout_panel_stats(heldout_df),
        "Held-out cohort metrics per configuration (mean +/- std across the 5 CV folds)",
        heldout_dir / "heldout_perfold_bar_chart.png",
    )
    for architecture, sub in heldout_df.groupby("architecture", sort=False):
        _plot_two_level_grouped_bar(
            sub["config"].tolist(),
            _heldout_panel_stats(sub),
            f"Held-out cohort metrics, {architecture} (mean +/- std across the 5 CV folds)",
            heldout_dir / f"arch_{slug(architecture)}_heldout_perfold_bar.png",
        )

    matrix_df = fold_auroc_wide.copy()
    matrix_df.columns = [f"fold_{c}" for c in matrix_df.columns]
    matrix_df["Mean"] = fold_auroc_wide.mean(axis=1)
    matrix_df["Std"] = fold_auroc_wide.std(axis=1, ddof=1)
    # fold_auroc_wide.pivot() sorted the index alphabetically; reindex to
    # architecture/MODEL_REGISTRY order (main_df's order) for the CSV and
    # figure -- this is a display reordering of a COPY, fold_auroc_wide
    # itself (feeding Friedman/Wilcoxon/wilcoxon_pmatrix_cv.csv below) is untouched.
    matrix_df = matrix_df.reindex(keys)
    matrix_df.to_csv(output_dir / "cv_fold_auroc_matrix.csv")
    _plot_fold_matrix(matrix_df, output_dir / "cv_fold_auroc_matrix.png")

    plot_friedman_ranks(
        fold_auroc_wide.T,
        friedman_stat,
        friedman_p,
        output_dir / "friedman_ranks_cv.png",
        title="Model rankings, CV validation-fold AUROC, frame-level",
    )
    plot_wilcoxon_pmatrix(
        wilcoxon_p,
        alpha,
        output_dir / "wilcoxon_pmatrix_cv.png",
        subtitle="paired on CV validation-fold AUROC, frame-level",
    )

    print(f"\nCV outputs written to {output_dir}")
    print(f"Held-out per-fold outputs written to {heldout_dir}")


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="ulcer_detection")
    p.add_argument("--mlflow-uri", default=cfg.mlflow_db)
    p.add_argument("--output-dir", default=str(cfg.results_cv_dir))
    p.add_argument(
        "--heldout-output-dir",
        default=str(cfg.results_heldout_dir),
        help="Where the held-out per-fold figures go (they describe the held-out cohort, not the CV folds)",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dry-run", action="store_true", help="Compute and print, skip writing files")
    p.add_argument(
        "--no-val-specificity",
        action="store_true",
        help=(
            "Skip reading the per-fold val probability artifacts. Faster and "
            "purely MLflow-side, but drops specificity/sensitivity from "
            "cv_per_fold_val_metrics.csv (no plotted figure is affected either way: "
            "the headline CV charts show AUROC only, see module docstring)."
        ),
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    if args.dry_run:
        print("[DRY RUN]")
    process_experiment(
        args.experiment,
        args.output_dir,
        alpha=args.alpha,
        dry_run=args.dry_run,
        with_val_specificity=not args.no_val_specificity,
        heldout_output_dir=args.heldout_output_dir,
    )
