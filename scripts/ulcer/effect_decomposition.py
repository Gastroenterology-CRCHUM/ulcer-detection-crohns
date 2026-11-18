"""
scripts/ulcer/effect_decomposition.py
--------------------------------------
Held-out 5-fold ensemble evaluation and pretraining effect decomposition.

For each of the 9 configurations, averages the 5 CV fold models' logged
held-out frame probabilities (no retraining, no re-inference) into an ensemble,
aggregates frames to clips (mean probability), and reports:

  1. Ensemble metrics, frame and clip level, with a patient-clustered 95% BCa
     bootstrap CI (19 held-out patients resampled; BCa because these metrics
     are not symmetrically distributed).
  2. Two pretraining effects over 5 architecture-matched pairs:
       method effect (Supervised -> DINOv1, ImageNet held constant)
       corpus effect (ImageNet -> GastroNet-5M, DINOv1 held constant)
     Primary: ensemble DeltaAUROC with its clustered BCa CI. Ensemble and
     best-fold DeLong (clip level, not clustered) are kept for traceability.
  3. The full pairwise bootstrap matrix (exploratory, uncorrected).
  4. A Friedman test on per-patient clip mean absolute error.
  5. The manuscript tables (results/ulcer/paper/).

Non-informative frames are excluded by default, as for training and CV data
(data/ulcer/heldout/informative_mask.csv); --no-filter-informative keeps them.

Usage
-----
    python -m scripts.ulcer.effect_decomposition
    python -m scripts.ulcer.effect_decomposition --dry-run
    python -m scripts.ulcer.effect_decomposition --n-bootstrap 20000 --seed 7
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
import numpy as np  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.config.paths import get_default_paths  # noqa: E402
from src.evaluation import heldout_plots as plots  # noqa: E402
from src.evaluation import heldout_tables as tables  # noqa: E402
from src.evaluation.heldout import (  # noqa: E402
    load_cohort,
    load_informative_mask,
    load_predictions,
)
from src.evaluation.mlflow_io import load_cv_runs  # noqa: E402
from src.evaluation.rank_tests import plot_friedman_ranks, run_friedman  # noqa: E402
from src.evaluation.reporting import write_table  # noqa: E402
from src.evaluation.style import slug  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")


def process_experiment(
    experiment_name: str,
    manifest_path: str,
    output_dir,
    paper_dir,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    dry_run: bool = False,
    filter_informative: bool = True,
    informative_mask_path: str | None = None,
) -> None:
    output_dir, paper_dir = Path(output_dir), Path(paper_dir)
    client = MlflowClient()

    # -- Cohort -------------------------------------------------------------
    print(f"Loading clip structure from {manifest_path} ...")
    full = load_cohort(manifest_path)
    frame_mask = None
    cohort = full
    if filter_informative:
        mask_path = informative_mask_path or str(
            Path(manifest_path).parent / "informative_mask.csv"
        )
        frame_mask = load_informative_mask(manifest_path, mask_path)
        n_kept = int(frame_mask.sum())
        print(
            f"  Informative-frame filter ON ({mask_path}): keeping {n_kept}/{len(frame_mask)} "
            f"frames ({len(frame_mask) - n_kept} excluded)."
        )
        cohort = load_cohort(manifest_path, frame_mask)
        n_dropped = len(full.clip_order) - len(cohort.clip_order)
        if n_dropped:
            print(f"  {n_dropped} clip(s) had zero surviving frames and were dropped.")
    n_patients = len(np.unique(cohort.clip_patient_ids))
    print(
        f"  {n_patients} patients, {len(cohort.clip_order)} clips, "
        f"{len(cohort.frame_clip_keys)} frames."
    )

    # -- Predictions --------------------------------------------------------
    runs = load_cv_runs(client, experiment_name)
    print(f"\nLoading heldout predictions for {len(runs)} configuration(s)...")
    configs = {}
    for model_key, model_runs in runs.items():
        print(f"  {model_key}: loading {len(model_runs.children)} fold(s)...")
        data = load_predictions(client, model_runs, full, cohort, frame_mask)
        if data is not None:
            configs[model_key] = data
    if not configs:
        print("No configurations could be loaded, aborting.")
        return

    # -- Tables -------------------------------------------------------------
    per_fold_df = tables.per_fold_table(configs)
    print(f"\nPer-fold table: {len(per_fold_df)} rows.")
    dispersion_df = tables.fold_dispersion_table(configs)
    print(f"Frame/clip fold-dispersion table: {len(dispersion_df)} configs.")

    print(f"\nBootstrapping ensemble metrics per config (n_bootstrap={n_bootstrap})...")
    ensemble = tables.ensemble_table(configs, cohort, n_bootstrap, seed, alpha)

    print(f"\nComputing effect decomposition for {len(tables.PAIRS)} pair(s)...")
    pair_df, pair_boots = tables.pair_table(configs, cohort, n_bootstrap, seed, alpha)

    print(f"\nBootstrapping full pairwise matrix, clip-level (n_bootstrap={n_bootstrap})...")
    matrix_df, matrix_boots = tables.pairwise_matrix(configs, cohort, n_bootstrap, seed, alpha)
    n_sig = int(matrix_df["significant"].sum())
    print(f"  {n_sig}/{len(matrix_df)} pairs significant (CI excludes 0)")

    print("Running Friedman on heldout clip mean absolute error (per-patient matrix)...")
    patient_error_df = tables.patient_error_table(configs, cohort)
    friedman_stat, friedman_p = run_friedman(patient_error_df)
    print(
        f"  {len(patient_error_df)} patients  Friedman chi2={friedman_stat:.4f}  p={friedman_p:.4f}"
    )

    paper = {
        "table1_heldout_ensemble": tables.paper_table1(ensemble),
        "tableS1_fold_dispersion": tables.paper_table_s1(configs),
        "text_comparisons": tables.paper_comparisons(matrix_boots, configs, cohort),
    }

    if dry_run:
        print("\n[DRY RUN] Skipping file writes.")
        for title, df in (
            ("Frame/clip fold-dispersion table", dispersion_df),
            ("Per-config table", ensemble.table),
            ("Effect decomposition", pair_df),
            ("Full pairwise bootstrap matrix (exploratory)", matrix_df),
            ("Per-patient mean absolute error", patient_error_df),
            *paper.items(),
        ):
            print(f"\n--- {title} ---\n{df.to_string(index=df is patient_error_df)}")
        return

    # -- Outputs ------------------------------------------------------------
    bootstrap_dir = output_dir / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    write_table(per_fold_df, output_dir / "per_fold_heldout_metrics")
    write_table(dispersion_df, output_dir / "heldout_fold_dispersion")
    plots.plot_fold_dispersion_table(dispersion_df, output_dir / "heldout_fold_dispersion.png")
    write_table(ensemble.table, output_dir / "heldout_ensemble_metrics")
    write_table(pair_df, output_dir / "effect_decomposition")
    matrix_df.to_csv(output_dir / "heldout_pairwise_bootstrap.csv", index=False)
    plots.plot_pairwise_matrix(
        list(configs), matrix_df, output_dir / "heldout_pairwise_bootstrap.png"
    )

    for model_key in configs:
        for metric in tables.METRICS:
            np.save(
                bootstrap_dir / f"{model_key}_{metric}.npy",
                ensemble.clip[model_key][metric]["replicates"],
            )
            np.save(
                bootstrap_dir / f"{model_key}_{metric}_frame.npy",
                ensemble.frame[model_key][metric]["replicates"],
            )
    for pair_id, result in pair_boots.items():
        np.save(bootstrap_dir / f"{pair_id}.npy", result["replicates"])

    plots.plot_ensemble_metrics(
        ensemble.table,
        "Heldout ensemble metrics per configuration (95% BCa CI, patient-clustered)",
        output_dir / "heldout_ensemble_metrics_bar.png",
    )
    if not pair_df.empty:
        plots.plot_forest(
            pair_df,
            "boot_ci_bca_lower",
            "boot_ci_bca_upper",
            "Effect decomposition, clip-level (95% BCa CI)",
            output_dir / "effect_decomposition_bca.png",
        )
        plots.plot_delong_pairs(pair_df, output_dir / "delong_pairs.png")
    for architecture, sub in ensemble.table.groupby("architecture", sort=False):
        plots.plot_ensemble_metrics(
            sub,
            f"Heldout ensemble metrics, {architecture} (95% BCa CI, patient-clustered)",
            output_dir / f"arch_{slug(architecture)}_heldout_metrics.png",
        )
    plots.plot_ensemble_roc(configs, cohort, output_dir / "roc_curves_ensemble.png")

    patient_error_df.to_csv(output_dir / "heldout_patient_mae.csv")
    plot_friedman_ranks(
        patient_error_df,
        friedman_stat,
        friedman_p,
        output_dir / "friedman_ranks_heldout_patient.png",
        title=(
            "Model rankings, heldout clip mean absolute error "
            f"({len(patient_error_df)} patients)"
        ),
        higher_is_better=False,
    )

    paper_dir.mkdir(parents=True, exist_ok=True)
    for name, df in paper.items():
        write_table(df, paper_dir / name)

    print(f"\nOutputs written to {output_dir} and {paper_dir}")


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="ulcer_detection")
    p.add_argument("--manifest", default=str(cfg.ulcer.heldout / "heldout_temporal_manifest.csv"))
    p.add_argument("--mlflow-uri", default=cfg.mlflow_db)
    p.add_argument("--output-dir", default=str(cfg.results_heldout_dir))
    p.add_argument("--paper-dir", default=str(cfg.results_ulcer_dir / "paper"))
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Compute and print, skip writing files")
    p.add_argument(
        "--filter-informative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude non-informative held-out frames before aggregation (default: on, as for "
            "training and CV data). Uses data/ulcer/heldout/informative_mask.csv; no "
            "re-inference needed, per-frame predictions do not depend on other frames."
        ),
    )
    p.add_argument(
        "--informative-mask",
        default=None,
        help="Informative-frame mask CSV (default: informative_mask.csv next to --manifest).",
    )
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
        args.paper_dir,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
        dry_run=args.dry_run,
        filter_informative=args.filter_informative,
        informative_mask_path=args.informative_mask,
    )
