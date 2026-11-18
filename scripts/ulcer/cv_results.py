"""
scripts/ulcer/cv_results.py
----------------------------
Cross-validation (training/validation-fold) results across the 9 model
configurations, from what run_modes.py logged to MLflow (no re-inference).

AUROC is the only metric featured on this side: sensitivity/specificity/F1 use a
threshold tuned on the same validation fold, so they are not a fair cross-model
comparison here. They are still recomputed and kept in cv_per_fold_val_metrics.csv
as diagnostic detail.

Usage
-----
    python -m scripts.ulcer.cv_results
    python -m scripts.ulcer.cv_results --dry-run
    python -m scripts.ulcer.cv_results --no-val-specificity
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.config.paths import get_default_paths  # noqa: E402
from src.evaluation import cv_plots as plots  # noqa: E402
from src.evaluation import cv_tables as tables  # noqa: E402
from src.evaluation.mlflow_io import load_cv_runs  # noqa: E402
from src.evaluation.rank_tests import (  # noqa: E402
    plot_friedman_ranks,
    plot_wilcoxon_pmatrix,
    run_friedman,
    run_wilcoxon_matrix,
)
from src.evaluation.reporting import write_table  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")


def _report_diagnostics(fold_df, diagnostics: tables.FoldDiagnostics) -> None:
    if diagnostics.artifact_misses:
        print(
            f"\n  [warn] {len(diagnostics.artifact_misses)} fold(s) had no readable val "
            f"probability artifact, excluded from val specificity: "
            f"{', '.join(diagnostics.artifact_misses)}"
        )
    if diagnostics.crosscheck_failures:
        print(
            f"\n  [ERROR] {len(diagnostics.crosscheck_failures)} fold(s) where recomputed "
            f"metrics disagree with the logged ones -- their specificity is "
            f"NOT trustworthy:"
        )
        for msg in diagnostics.crosscheck_failures:
            print(f"    - {msg}")
    else:
        n_ok = int(fold_df["val_specificity"].notna().sum())
        print(
            f"\n  Val specificity recomputed for {n_ok}/{len(fold_df)} folds "
            f"(cv_per_fold_val_metrics.csv); recomputed sensitivity/F1/AUROC "
            f"match the logged values on all of them."
        )


def process_experiment(
    experiment_name: str,
    output_dir,
    alpha: float = 0.05,
    dry_run: bool = False,
    with_val_specificity: bool = True,
) -> None:
    output_dir = Path(output_dir)
    client = MlflowClient()

    print("Loading CV runs...")
    runs = load_cv_runs(client, experiment_name)
    if not runs:
        print("No CV configurations could be loaded, aborting.")
        return

    main_df = tables.main_table(runs)
    fold_df, diagnostics = tables.per_fold_table(runs, with_val_specificity)
    if with_val_specificity:
        _report_diagnostics(fold_df, diagnostics)

    print(f"\nMain table: {len(main_df)} configs, per-fold table: {len(fold_df)} rows.")
    print(main_df.to_string(index=False))

    # Friedman + Wilcoxon on validation-fold AUROC, the threshold-free metric.
    fold_auroc_wide = fold_df.pivot(index="config", columns="fold", values="val_auroc")
    friedman_stat, friedman_p = run_friedman(fold_auroc_wide.T)
    wilcoxon_p = run_wilcoxon_matrix(fold_auroc_wide)
    print(f"\nFriedman (CV val AUROC): chi2={friedman_stat:.4f}  p={friedman_p:.4f}")

    if dry_run:
        print("\n[DRY RUN] Skipping file writes.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(main_df, output_dir / "cv_main_table")
    fold_df.to_csv(output_dir / "cv_per_fold_val_metrics.csv", index=False)
    wilcoxon_p.to_csv(output_dir / "wilcoxon_pmatrix_cv.csv")

    keys = main_df["config"].tolist()
    plots.plot_main_table(main_df, output_dir / "cv_main_table.png")
    plots.plot_auroc_legend_bar(
        keys,
        main_df["val_auroc_mean"].tolist(),
        main_df["val_auroc_std"].tolist(),
        "CV validation-fold AUROC per configuration, frame-level",
        output_dir / "cv_bar_chart.png",
        ref_line=0.5,
        ref_label="0.5 (chance)",
    )
    plots.plot_model_bar(
        keys,
        main_df["threshold_mean"].tolist(),
        main_df["threshold_std"].tolist(),
        "Optimal frame threshold (mean +/- std across folds)",
        "CV threshold dispersion across folds, frame-level",
        output_dir / "cv_threshold_dispersion.png",
        ref_line=0.5,
        ref_label="0.5 (default)",
    )

    # pivot() sorted configs alphabetically; the matrix is re-ordered for display
    # only, Friedman/Wilcoxon above use fold_auroc_wide as is.
    matrix_df = tables.fold_auroc_matrix(fold_auroc_wide, keys)
    matrix_df.to_csv(output_dir / "cv_fold_auroc_matrix.csv")
    plots.plot_fold_matrix(matrix_df, output_dir / "cv_fold_auroc_matrix.png")

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


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="ulcer_detection")
    p.add_argument("--mlflow-uri", default=cfg.mlflow_db)
    p.add_argument("--output-dir", default=str(cfg.results_cv_dir))
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
    )
