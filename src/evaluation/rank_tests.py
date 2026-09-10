"""
src/evaluation/rank_tests.py
----------------------------
Friedman + Wilcoxon rank comparison across models, and Holm-Bonferroni
multiple-comparison correction. Pure computation/plotting, no MLflow, no
file I/O beyond the figure itself.

Two DIFFERENT DataFrame conventions are in play here, matching what each
test naturally needs -- keep them straight when building the input frame:

  Friedman family (run_friedman, plot_friedman_ranks):
      ROWS = blocks (folds, patients, ...); COLUMNS = models (the treatment
      being compared). This is the standard textbook Friedman layout, and
      what scipy.stats.friedmanchisquare(*samples) naturally wants when each
      *sample* is one column's values across the block rows.

  Wilcoxon family (run_wilcoxon_matrix, plot_wilcoxon_pmatrix):
      ROWS = models; COLUMNS = blocks (folds). Pairwise-compares two model
      ROWS' values across the shared blocks.

Both keep raw MODEL_REGISTRY config strings as their model-identifying
index/columns (config_color and the config_label default both raise on
anything else).

Public API
----------
run_friedman(df)          -> (chi2, p)  -- omnibus test on the (blocks x models)
                                            DataFrame's COLUMNS, blocked on rows
run_wilcoxon_matrix(df)   -> DataFrame of pairwise p-values, on a (models x blocks) DataFrame
holm_bonferroni(pvals)    -> np.ndarray of adjusted p-values
plot_friedman_ranks(df, stat, p, output_path, title, *, higher_is_better=True, label_fn=config_label)
    df: (blocks x models), same convention as run_friedman.
plot_wilcoxon_pmatrix(p_df, alpha, output_path, subtitle="", *, label_fn=config_label)
    label_fn maps a raw config key to display text for tick labels only --
    df/p_df (and anything a caller writes to CSV from them) keep raw keys.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from src.evaluation.style import (
    STATUS_NEUTRAL,
    STATUS_NOT_SIGNIFICANT,
    STATUS_SIGNIFICANT,
    config_color,
    config_label,
)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def run_friedman(df: pd.DataFrame) -> tuple[float, float]:
    """Friedman omnibus test on a (blocks x models) DataFrame -- tests
    whether the COLUMNS (the treatments being compared, e.g. models) differ,
    blocked on the ROWS (e.g. folds or patients). Each column is passed to
    scipy's friedmanchisquare as one treatment's repeated measurements
    across blocks."""
    model_columns = [df[col].values for col in df.columns]
    stat, p = friedmanchisquare(*model_columns)
    return float(stat), float(p)


def run_wilcoxon_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank test (paired by fold), no correction."""
    models = df.index.tolist()
    n = len(models)
    p_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = df.iloc[i].values, df.iloc[j].values
            p_val = 1.0 if np.allclose(a, b) else wilcoxon(a, b)[1]
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
    return pd.DataFrame(p_matrix, index=models, columns=models)


def holm_bonferroni(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values, in the same order as the input."""
    pvals_arr = np.asarray(pvals, dtype=float)
    n = len(pvals_arr)
    order = np.argsort(pvals_arr)
    sorted_p = pvals_arr[order]
    multipliers = n - np.arange(n)
    step = np.minimum(sorted_p * multipliers, 1.0)
    adjusted_sorted = np.maximum.accumulate(step)
    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted
    return adjusted


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_friedman_ranks(
    df: pd.DataFrame,
    stat: float,
    p: float,
    output_path,
    title: str = "Model rankings",
    *,
    higher_is_better: bool = True,
    label_fn: Callable[[str], str] = config_label,
) -> None:
    """df: (blocks x models), same convention as run_friedman. Ranks the
    MODEL COLUMNS within each block ROW (axis=1) -- rank 1 = best in that
    block -- then averages each model's rank across blocks (axis=0).

    higher_is_better: set False for an error-type metric (e.g. mean absolute
    error, log-loss) where the SMALLEST value should rank 1st.

    label_fn maps a raw MODEL_REGISTRY key (df's column values) to a display
    string for the y-axis; config_color always keys on the RAW value, so
    color and label can never desync even though the tick text is mapped."""
    ranked = df.rank(axis=1, ascending=not higher_is_better)
    mean_ranks = ranked.mean(axis=0).sort_values()
    sig = p < 0.05
    sig_color = STATUS_SIGNIFICANT if sig else STATUS_NEUTRAL

    fig, ax = plt.subplots(figsize=(max(8, len(mean_ranks) * 0.9), 5))
    y = np.arange(len(mean_ranks))
    bars = ax.barh(y, mean_ranks.values, color=[config_color(m) for m in mean_ranks.index])
    ax.set_yticks(y)
    ax.set_yticklabels([label_fn(m) for m in mean_ranks.index])
    ax.set_xlabel("Mean rank (1 = best)")
    ax.set_title(title, fontsize=11)
    ax.text(
        0.5, 1.06,
        f"Friedman chi2={stat:.2f}, p={p:.4f}  ({'SIGNIFICANT' if sig else 'not significant'})",
        transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold", color=sig_color,
    )
    ax.invert_yaxis()
    for bar, val in zip(bars, mean_ranks.values):
        ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, mean_ranks.max() + 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wilcoxon_pmatrix(
    p_df: pd.DataFrame,
    alpha: float,
    output_path,
    subtitle: str = "",
    *,
    label_fn: Callable[[str], str] = config_label,
) -> None:
    """Each cell = Wilcoxon signed-rank p-value for row-model vs. column-model.
    Diagonal is self-comparison. Categorical significant/not-significant
    coloring (not a continuous gradient) so the reader doesn't have to map
    color to a p-value by eye. label_fn maps p_df's raw index/columns to
    display strings for the tick labels only -- p_df itself (and whatever a
    caller writes to CSV from it) keeps the raw MODEL_REGISTRY keys."""
    n = len(p_df)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.0), max(7, n * 0.9)))

    import matplotlib.colors as mcolors

    colors = np.ones((n, n, 4))
    for i in range(n):
        for j in range(n):
            if i == j:
                colors[i, j] = mcolors.to_rgba("#e1e0d9")
            else:
                p = p_df.iloc[i, j]
                colors[i, j] = mcolors.to_rgba(STATUS_SIGNIFICANT if p < alpha else STATUS_NOT_SIGNIFICANT)
    ax.imshow(colors, aspect="auto")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", color=STATUS_NEUTRAL, fontsize=9)
                continue
            p = p_df.iloc[i, j]
            txt = f"{p:.3f}" if p >= 0.001 else "<0.001"
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=8,
                fontweight="bold" if p < alpha else "normal", color="#0b0b0b",
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels([label_fn(c) for c in p_df.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label_fn(i) for i in p_df.index], fontsize=8)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    title = "Wilcoxon signed-rank test: row model vs. column model"
    if subtitle:
        title += f"\n{subtitle}"
    title += f" (alpha={alpha})"
    ax.set_title(title, fontsize=11)
    legend_elems = [
        mpatches.Patch(facecolor=STATUS_SIGNIFICANT, label=f"Significant (p < {alpha})"),
        mpatches.Patch(facecolor=STATUS_NOT_SIGNIFICANT, label=f"Not significant (p >= {alpha})"),
    ]
    ax.legend(
        handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=2, fontsize=8, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
