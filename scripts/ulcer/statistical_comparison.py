"""
scripts/ulcer/statistical_comparison.py
----------------------------------------
Friedman + Nemenyi post-hoc and Wilcoxon pairwise tests on per-fold
validation AUROCs across the 9 model configurations.

Reads fold-level AUROC metrics logged to MLflow (metric key:
``fold_val_auroc``, stepped by fold index 0–4) for all top-level runs in the
``ulcer_detection`` experiment.  Produces two figures:

    results/ulcer/cv/friedman_ranks.png     — mean rank per model + Friedman p
    results/ulcer/cv/wilcoxon_pmatrix.png   — Wilcoxon signed-rank p-value matrix

Usage
-----
    # After training all 9 models:
    python -m scripts.ulcer.statistical_comparison

    # Point at a custom MLflow store:
    python -m scripts.ulcer.statistical_comparison \\
        --mlflow-uri sqlite:///mlflow.db \\
        --experiment ulcer_detection \\
        --output-dir results/ulcer/cv

Requirements
------------
    scikit-posthocs >= 0.9.0   (add to pyproject.toml / pip install)
    mlflow, scipy, matplotlib, seaborn, pandas, numpy
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("muted")


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def _load_fold_aurocs(
    mlflow_uri: str,
    experiment_name: str,
    metric_key: str = "fold_val_auroc",
    n_folds: int = 5,
) -> pd.DataFrame:
    """Return a (n_models × n_folds) DataFrame of per-fold AUROCs.

    Rows = model name (run name), columns = fold indices 0 … n_folds-1.
    Only top-level runs (no ``mlflow.parentRunId`` tag) are included.
    """
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(mlflow_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found in MLflow store at {mlflow_uri!r}."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        max_results=500,
    )
    # Keep only top-level runs (not CV fold children)
    top_level = [r for r in runs if not r.data.tags.get("mlflow.parentRunId")]

    records: dict[str, list[float]] = {}
    for run in top_level:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        history = client.get_metric_history(run.info.run_id, metric_key)
        if not history:
            warnings.warn(
                f"Run '{run_name}' has no metric '{metric_key}' — skipping.",
                stacklevel=2,
            )
            continue
        # history is ordered by step; collect up to n_folds values
        fold_aurocs = [m.value for m in sorted(history, key=lambda m: m.step)][:n_folds]
        if len(fold_aurocs) < n_folds:
            warnings.warn(
                f"Run '{run_name}' has only {len(fold_aurocs)}/{n_folds} fold values — skipping.",
                stacklevel=2,
            )
            continue
        records[run_name] = fold_aurocs

    if len(records) < 2:
        raise RuntimeError(
            f"Need ≥ 2 models with {n_folds} folds of '{metric_key}' to run comparisons. "
            f"Found {len(records)}."
        )

    df = pd.DataFrame(records, index=list(range(n_folds))).T  # (models × folds)
    df.index.name = "model"
    df.columns = [f"fold_{i}" for i in range(n_folds)]
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def run_friedman(df: pd.DataFrame) -> tuple[float, float]:
    """Friedman test on (models × folds) DataFrame.

    Returns (statistic, p_value).
    """
    fold_columns = [df[col].values for col in df.columns]
    stat, p = friedmanchisquare(*fold_columns)
    return float(stat), float(p)


def run_nemenyi(df: pd.DataFrame) -> pd.DataFrame:
    """Nemenyi post-hoc test; returns a symmetric p-value DataFrame."""
    return sp.posthoc_nemenyi_friedman(df.values)


def run_wilcoxon_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank test (no correction).

    Returns a symmetric DataFrame of p-values indexed and columned by model name.
    """
    models = df.index.tolist()
    n = len(models)
    p_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = df.iloc[i].values
            b = df.iloc[j].values
            if np.allclose(a, b):
                p_val = 1.0
            else:
                _, p_val = wilcoxon(a, b)
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
    return pd.DataFrame(p_matrix, index=models, columns=models)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_friedman_ranks(
    df: pd.DataFrame,
    friedman_stat: float,
    friedman_p: float,
    output_path: Path,
) -> None:
    """Bar chart of mean rank per model with Friedman p annotated."""
    # Rank within each fold (ascending = lower AUROC = higher rank number → worse)
    # Use descending rank so rank 1 = best
    ranked = df.rank(axis=0, ascending=False)  # rank across models for each fold
    mean_ranks = ranked.mean(axis=1).sort_values()

    fig, ax = plt.subplots(figsize=(max(8, len(mean_ranks) * 0.9), 5))
    bars = ax.barh(mean_ranks.index, mean_ranks.values, color=sns.color_palette("muted", len(mean_ranks)))
    ax.set_xlabel("Mean rank (1 = best)")
    ax.set_title(
        f"Model rankings — 5-fold CV validation AUROC\n"
        f"Friedman χ²={friedman_stat:.2f}, p={friedman_p:.4f}"
        + (" *" if friedman_p < 0.05 else " (n.s.)"),
        fontsize=11,
    )
    ax.invert_yaxis()
    for bar, val in zip(bars, mean_ranks.values):
        ax.text(
            val + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, mean_ranks.max() + 0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_wilcoxon_pmatrix(
    p_df: pd.DataFrame,
    alpha: float,
    output_path: Path,
) -> None:
    """Heatmap of Wilcoxon pairwise p-values with significance overlay."""
    fig, ax = plt.subplots(figsize=(max(7, len(p_df) * 0.9), max(6, len(p_df) * 0.8)))
    mask = np.eye(len(p_df), dtype=bool)  # hide diagonal

    annot = p_df.applymap(lambda v: f"{v:.3f}" if not np.isnan(v) else "")

    sns.heatmap(
        p_df,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "p-value"},
    )
    # Overlay asterisks for significant pairs
    for i in range(len(p_df)):
        for j in range(len(p_df)):
            if i != j and p_df.iloc[i, j] < alpha:
                ax.text(
                    j + 0.5,
                    i + 0.75,
                    "*",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                )

    ax.set_title(
        f"Wilcoxon signed-rank pairwise p-values (α={alpha})\n"
        "* = significant after correction",
        fontsize=11,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Friedman + Wilcoxon statistical comparison of CV fold AUROCs."
    )
    p.add_argument(
        "--mlflow-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    p.add_argument(
        "--experiment",
        default="ulcer_detection",
        help="MLflow experiment name (default: ulcer_detection)",
    )
    p.add_argument(
        "--metric",
        default="fold_val_auroc",
        help="Per-fold metric key logged to MLflow (default: fold_val_auroc)",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for Wilcoxon overlay (default: 0.05)",
    )
    p.add_argument(
        "--output-dir",
        default="results/ulcer/cv",
        help="Directory for output figures (default: results/ulcer/cv)",
    )
    return p


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fold AUROCs from MLflow ({args.mlflow_uri}) …")
    df = _load_fold_aurocs(
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        metric_key=args.metric,
        n_folds=args.n_folds,
    )
    print(f"  Loaded {len(df)} models × {len(df.columns)} folds")
    print(df.to_string())

    print("\nRunning Friedman test …")
    stat, p = run_friedman(df)
    print(f"  χ² = {stat:.4f},  p = {p:.4f}" + (" *" if p < args.alpha else " (n.s.)"))

    print("\nPlotting Friedman ranks …")
    plot_friedman_ranks(
        df,
        friedman_stat=stat,
        friedman_p=p,
        output_path=output_dir / "friedman_ranks.png",
    )

    print("\nRunning Wilcoxon pairwise tests …")
    p_matrix = run_wilcoxon_matrix(df)
    # Re-index so model names match
    p_matrix.index = df.index
    p_matrix.columns = df.index

    print("\nPlotting Wilcoxon p-value matrix …")
    plot_wilcoxon_pmatrix(
        p_matrix,
        alpha=args.alpha,
        output_path=output_dir / "wilcoxon_pmatrix.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
