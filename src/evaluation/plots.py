"""
src/evaluation/plots.py
-----------------------
Pure visualisation, no MLflow, no file management, no metric computation.

Every function returns the matplotlib Figure so the caller decides
what to do with it (save, log to MLflow, display in a notebook…).

Public API
----------
plot_roc_curve(labels, probs, threshold, title)         -> Figure
plot_confusion_matrix(cm, threshold, class_names)       -> Figure
plot_roc_curves(roc_data, title, *, ax=None)            -> Figure
plot_delong_heatmap(p_matrix, df_summary, alpha)        -> Figure
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.style import STATUS_NEUTRAL, STATUS_NOT_SIGNIFICANT, STATUS_SIGNIFICANT

# ---------------------------------------------------------------------------
# Single-model ROC curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    name: str,
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
) -> plt.Figure:
    """Plot a single ROC curve with an optional operating-point marker.

    Args:
        name:      Model name.
        labels:    Ground-truth binary labels.
        probs:     Predicted probabilities for the positive class.
                   Accepts shape (N,) or (N, 2), second column is used.
        threshold: If provided, marks the closest point on the curve in red.
        title:     Figure title.

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        probs = probs[:, 1]

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"ROC Curve, Ulcer detection with {name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if threshold is not None:
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(
            fpr[idx],
            tpr[idx],
            color="red",
            zorder=5,
            label=f"Threshold = {threshold:.2f}",
        )
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    threshold: float,
    class_names: tuple[str, str] = ("No ulcer", "Ulcer"),
    title: str = "Confusion matrix, Ulcer detection",
) -> plt.Figure:
    """Plot a colour-coded normalised confusion matrix.

    Cell colours:
    - Dark blue → high rate (row-normalised)
    - White → low rate
    - Grey → zero samples (avoid misleading white cells)

    Args:
        cm:          2x2 confusion matrix (sklearn convention: rows = true,
                     columns = predicted).
        threshold:   Decision threshold shown in the title.
        class_names: (negative_label, positive_label).
        title:       Figure title.

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    # Row-normalised rates + raw counts for annotations
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)  # Avoid division by zero

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + f"  (threshold = {threshold:.2f})")

    fig.tight_layout()
    return fig


def plot_confusion_matrix_multiclass(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion matrix",
) -> plt.Figure:
    """Normalised confusion matrix for N classes.

    Args:
        cm:          NxN confusion matrix (sklearn convention).
        class_names: List of class labels (length N).
        title:       Figure title.

    Returns:
        matplotlib Figure.
    """
    n = len(class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(n * 2, n * 1.8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-model ROC curves
# ---------------------------------------------------------------------------


def plot_roc_curves(
    roc_data: list[dict],
    title: str = "ROC curves, Ulcer detection",
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Overlay ROC curves for several models on the same axes.

    Args:
        roc_data: List of dicts, each with keys:
                    "name"  (str)   - label shown in the legend,
                    "fpr"   (array) - false positive rates,
                    "tpr"   (array) - true positive rates,
                    "auc"   (float) - AUROC value,
                    "color" (str, optional) - explicit line color; omit to use
                            matplotlib's default cycle.
        title:    Panel/figure title.
        ax:       Draw onto this existing Axes instead of creating a new
                  Figure -- e.g. to place two ROC panels (frame/clip level)
                  side by side in one figure. The legend is still drawn on
                  `ax`; the CALLER owns fig-level layout (tight_layout /
                  savefig) in that case. Omit for the original one-panel
                  behavior (a new Figure is created and fully laid out here).

    Returns:
        The Figure `ax` belongs to (a new one-axes Figure when ax is None).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        owns_figure = True
    else:
        fig = ax.figure
        owns_figure = False

    for entry in roc_data:
        ax.plot(
            entry["fpr"],
            entry["tpr"],
            linewidth=2,
            color=entry.get("color"),
            label=f"{entry['name']}  (AUC = {entry['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1 − Specificity  (FPR)")
    ax.set_ylabel("Sensitivity  (TPR)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    n_series = len(roc_data) + 1
    ncol = 2 if n_series > 5 else 1
    if owns_figure:
        fig.tight_layout(rect=(0, 0.20, 1, 1))
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, fontsize=9, frameon=True
        )
    else:
        # Each panel keeps its OWN legend (AUC differs per panel, e.g. frame
        # vs clip level, so a single shared legend can't show both) -- the
        # caller reserves bottom margin for it via its own tight_layout rect.
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol, fontsize=8, frameon=True
        )
    return fig


# ---------------------------------------------------------------------------
# DeLong heatmap
# ---------------------------------------------------------------------------


def plot_delong_heatmap(
    p_matrix: pd.DataFrame,
    df_summary: pd.DataFrame,
    alpha: float = 0.05,
    title: str | None = None,
) -> plt.Figure:
    """Visualise DeLong p-value matrix and pairwise comparison table.

    Colour coding (categorical, not a gradient -- consistent with every other
    significance display in this project, see src/evaluation/style.py and
    src/evaluation/rank_tests.py::plot_wilcoxon_pmatrix):
      - Green (STATUS_SIGNIFICANT)     -> p < alpha
      - Pale red (STATUS_NOT_SIGNIFICANT) -> p >= alpha
      - Light gray -> diagonal / lower triangle (self-comparison, not shown)

    Args:
        p_matrix:   N×N DataFrame of p-values (output of delong_matrix).
        df_summary: Pairwise comparison DataFrame (output of delong_matrix).
        alpha:      Significance threshold.
        title:      Optional title override; omit for the default
                    "DeLong test p-values (α = ...)".

    Returns:
        matplotlib Figure (not yet saved or shown).
    """
    names = p_matrix.index.tolist()
    n = len(names)
    max_name_len = max((len(str(name)) for name in names), default=8)
    fig_width = max(14, 9 + max_name_len * 0.35)
    n_pairs = len(df_summary)
    fig_height = max(5, n * 0.5, 1.3 + n_pairs * 0.22)

    fig, (ax_heat, ax_table) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # ── Heatmap ──────────────────────────────────────────────────────────────
    display = p_matrix.copy().astype(float)
    for i in range(n):
        for j in range(i + 1):  # lower triangle + diagonal -> NaN
            display.iloc[i, j] = np.nan

    diag_gray = mcolors.to_rgba("#e1e0d9")
    sig_color = mcolors.to_rgba(STATUS_SIGNIFICANT)
    not_sig_color = mcolors.to_rgba(STATUS_NOT_SIGNIFICANT)
    colors = np.ones((n, n, 4))
    for i in range(n):
        for j in range(n):
            if i >= j:
                colors[i, j] = diag_gray
            else:
                p = display.iloc[i, j]
                colors[i, j] = sig_color if p < alpha else not_sig_color

    ax_heat.imshow(colors, aspect="auto")

    for i in range(n):
        for j in range(i + 1, n):
            p = display.iloc[i, j]
            txt = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax_heat.text(
                j, i, txt, ha="center", va="center", fontsize=8,
                fontweight="bold" if p < alpha else "normal", color="#0b0b0b",
            )
        ax_heat.text(i, i, "-", ha="center", va="center", fontsize=10, color=STATUS_NEUTRAL)

    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(names, fontsize=9)
    ax_heat.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.5)
    ax_heat.tick_params(which="minor", length=0)
    ax_heat.set_title(title or f"DeLong test p-values  (alpha = {alpha})", fontsize=11, pad=10)

    legend_elems = [
        mpatches.Patch(facecolor=STATUS_SIGNIFICANT, label=f"Significant  (p < {alpha})"),
        mpatches.Patch(facecolor=STATUS_NOT_SIGNIFICANT, label=f"Not significant  (p >= {alpha})"),
    ]
    ax_heat.legend(
        handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, fontsize=8, frameon=False,
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    ax_table.axis("off")

    y = 0.97
    ax_table.text(
        0.05, y, "DeLong pairwise comparisons", fontsize=11, fontweight="bold",
        transform=ax_table.transAxes, va="top",
    )
    y -= 0.06

    headers = ["Model A", "Model B", "AUC delta", "z", "p-value", "sig."]
    # Model A/B each get width proportional to the longest name actually present,
    # instead of a fixed fraction sized for short names like "fold_1"/"ensemble".
    name_col_width = min(0.30, 0.09 + max_name_len * 0.011)
    col_x = [
        0.00,
        name_col_width,
        2 * name_col_width,
        2 * name_col_width + 0.14,
        2 * name_col_width + 0.27,
        2 * name_col_width + 0.40,
    ]

    for hdr, x in zip(headers, col_x):
        ax_table.text(
            x + 0.02, y, hdr, color=STATUS_NEUTRAL, fontsize=8, fontweight="bold",
            transform=ax_table.transAxes, va="top",
        )
    y -= 0.04
    ax_table.axhline(y, color="#e1e0d9", linewidth=0.8, xmin=0.02, xmax=0.98)
    y -= 0.01

    row_step = y / max(n_pairs, 1)
    for _, row in df_summary.iterrows():
        color = STATUS_SIGNIFICANT if row["significant"] else "#0b0b0b"
        vals = [
            str(row["Model A"])[:28],
            str(row["Model B"])[:28],
            f"{row['ΔAUC']:+.4f}",
            f"{row['z']:.2f}",
            f"{row['p-value']:.4f}" if row["p-value"] >= 0.001 else "<0.001",
            "Yes" if row["significant"] else "No",
        ]
        for val, x in zip(vals, col_x):
            ax_table.text(
                x + 0.02, y, val, color=color, fontsize=7.5, transform=ax_table.transAxes, va="top"
            )
        y -= row_step

    fig.tight_layout(pad=1.5)
    return fig


# ----------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------


def plot_learning_curves(
    results_df: pd.DataFrame,
    metric: str = "f1",
    title: str = "Data efficiency, learning curves",
) -> plt.Figure:

    head_types = results_df["head_type"].unique()
    n_heads = len(head_types)

    fig, axes = plt.subplots(1, n_heads, figsize=(6 * n_heads, 5), sharey=True)
    if n_heads == 1:
        axes = [axes]

    for ax, head in zip(axes, head_types):
        sub = results_df[results_df["head_type"] == head]
        for model_name, grp in sub.groupby("model"):
            grp = grp.sort_values("subset_ratio")
            x = grp["subset_ratio"] * 100
            y = grp[f"{metric}_mean"]
            # Ensure the std column exists (from the np.std we added earlier)
            y_std = grp[f"{metric}_std"] if f"{metric}_std" in grp.columns else 0

            # 1. Plot the main line
            (line,) = ax.plot(x, y, marker="o", label=model_name, linewidth=2)

            # 2. Add the shadow (Standard Deviation)
            # Use the same color as the line with low alpha (transparency)
            ax.fill_between(x, y - y_std, y + y_std, color=line.get_color(), alpha=0.15)

        ax.set_title(f"Head : {head}", fontweight="bold")
        ax.set_xlabel("Training set size (%)")
        ax.set_ylabel(metric.upper() if ax == axes[0] else "")
        ax.set_xticks([10, 25, 50, 75, 100])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
