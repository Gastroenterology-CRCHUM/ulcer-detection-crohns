"""
src/evaluation/cv_plots.py
--------------------------
Figures of the cross-validation (validation-fold) report. Callers set the
matplotlib backend/style.

Public API
----------
plot_model_bar(keys, means, stds, ylabel, title, output_path, *, ref_line, ref_label)
plot_auroc_legend_bar(keys, means, stds, title, output_path, *, ref_line, ref_label)
plot_fold_matrix(matrix_df, output_path)
plot_main_table(main_df, output_path)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.plot_tables import column_widths, draw_config_table, text_lengths
from src.evaluation.style import (
    STATUS_NEUTRAL,
    config_color,
    config_label,
    config_labels,
    metric_label,
)


def _draw_grouped_bars(ax, keys, labels, metric_values: dict, metric_stds: dict, xlabels) -> None:
    """Draw one grouped-bar panel: metrics on the x-axis, one bar per model,
    error bars = std across folds. Used by the held-out two-level chart.

    keys/labels are separate parameters, never one list: config_color raises
    on anything that isn't a raw MODEL_REGISTRY key, so the color lookup
    always uses `keys` while the legend text uses `labels`.
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
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#4a4a4a"},
        )
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=0)

    ax.set_ylim(0, 1.05)


def plot_model_bar(
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


def plot_auroc_legend_bar(
    keys, means, stds, title, output_path, *, ref_line=None, ref_label=None
) -> None:
    """One tightly-packed cluster of bars (all models in a single 'AUROC'
    x-slot)"""
    labels = [f"{config_label(k)}  ({m:.3f} +/- {s:.3f})" for k, m, s in zip(keys, means, stds)]
    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_grouped_bars(
        ax, keys, labels, {"auroc": list(means)}, {"auroc": list(stds)}, [metric_label("auroc")]
    )
    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="#c3c2b7", linewidth=1, label=ref_label)
    ax.set_ylabel("Validation-fold AUROC (mean +/- std)", fontsize=18)
    ax.set_title(title, fontsize=24, fontweight="bold")
    ax.tick_params(labelsize=16)
    fig.tight_layout(rect=(0, 0.30, 1, 1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=15, frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fold_matrix(df: pd.DataFrame, output_path) -> None:
    """Fold x model AUROC as a plain table, row order = architecture."""
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

    lengths = text_lengths([header] + cell_text)
    fig_w = max(9.0, 1.2 + 0.145 * sum(lengths))
    fig_h = max(3.2, 0.55 + 0.42 * len(cell_text))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    table = draw_config_table(
        ax, header, cell_text, df.index, column_widths(lengths), fontsize=8.5, row_scale=1.5
    )
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


def plot_main_table(df: pd.DataFrame, output_path) -> None:
    """Render the main CV comparison table as an image (not just CSV/MD) so
    there's a single figure a reader can drop straight into a slide/manuscript.

    AUROC only (mean +/- std): the one threshold-free validation-fold metric,
    see module docstring. Sensitivity/specificity/F1/precision stay out of
    this headline table.
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

    lengths = text_lengths([header] + cell_text)
    fig_w = max(11.0, 1.6 + 0.11 * sum(lengths))

    fig, ax = plt.subplots(figsize=(fig_w, 0.5 + 0.4 * len(df)))
    draw_config_table(
        ax, header, cell_text, df["config"], column_widths(lengths), fontsize=8.5, row_scale=1.5
    )
    ax.set_title(
        "CV main table: validation-fold AUROC (mean +/- std) and best fold",
        fontsize=11,
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
