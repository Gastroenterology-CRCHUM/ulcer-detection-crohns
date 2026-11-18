"""
scripts/ulcer/log_heldout_clip_metrics.py
-----------------------------------------
Post-hoc clip-level held-out metrics for every CV fold, logged back to MLflow.

For each CV parent run, loads each fold's held-out frame predictions, averages
them per clip, applies that fold's clip threshold, and logs:

  Child run  -> heldout_clip__{metric}_mean   (per-fold clip metrics)
  Parent run -> cv_mean_heldout_clip_{metric}  (mean across folds)
                cv_std_heldout_clip_{metric}   (std  across folds)

All held-out frames are used (no informative-frame filter), as in the values
already logged; the filtered analysis is effect_decomposition.py.

Usage
-----
    python -m scripts.ulcer.log_heldout_clip_metrics
    python -m scripts.ulcer.log_heldout_clip_metrics --dry-run
    python -m scripts.ulcer.log_heldout_clip_metrics --force
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.config.paths import get_default_paths  # noqa: E402
from src.evaluation.heldout import aggregate_to_clip, load_cohort, point_metrics  # noqa: E402
from src.evaluation.mlflow_io import artifact_paths, download_npy, load_cv_runs  # noqa: E402

FOLD_METRICS = ("auroc", "accuracy", "f1", "precision", "sensitivity", "specificity")


def process_experiment(
    experiment_name: str, manifest_path: str, dry_run: bool = False, force: bool = False
) -> None:
    client = MlflowClient()
    cohort = load_cohort(manifest_path)
    runs = load_cv_runs(client, experiment_name)
    print(f"Found {len(runs)} CV parent run(s) in '{experiment_name}'.")

    for model_key, (parent, children) in runs.items():
        if not force and "cv_mean_heldout_clip_auroc" in parent.data.metrics:
            print(
                f"  {model_key}: clip heldout metrics already present, skip "
                f"(use --force to overwrite)."
            )
            continue

        print(f"  {model_key}: processing {len(children)} fold(s)...")
        fold_values: dict[str, list[float]] = {k: [] for k in FOLD_METRICS}
        for child in children:
            name = child.data.tags.get("mlflow.runName", child.info.run_id[:8])
            paths = artifact_paths(client, child.info.run_id, "predictions/heldout")
            probs_path = next((p for p in paths if "probs" in p), None)
            if probs_path is None:
                print(f"    {name}: no heldout probs artifact, skip.")
                continue

            probs = download_npy(client, child.info.run_id, probs_path)
            clip_probs = aggregate_to_clip(probs, cohort.frame_clip_keys, cohort.clip_order)
            threshold = child.data.metrics.get("fold_optimal_clip_threshold", 0.5)
            m = point_metrics(cohort.clip_labels, clip_probs, threshold)
            n_clips = len(cohort.clip_order)
            print(
                f"    {name}: AUROC={m['auroc']:.3f}  F1={m['f1']:.3f}  "
                f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  n_clips={n_clips}"
            )

            if not dry_run:
                with mlflow.start_run(run_id=child.info.run_id):
                    mlflow.log_metrics(
                        {f"heldout_clip__{k}_mean": v for k, v in m.items() if not np.isnan(v)}
                    )
                    mlflow.log_metric("heldout_clip_n_clips", n_clips)

            for k in FOLD_METRICS:
                if not np.isnan(m[k]):
                    fold_values[k].append(m[k])

        parent_log: dict[str, float] = {}
        for k, vals in fold_values.items():
            if not vals:
                continue
            arr = np.array(vals)
            parent_log[f"cv_mean_heldout_clip_{k}"] = float(arr.mean())
            parent_log[f"cv_std_heldout_clip_{k}"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        print(
            f"    -> parent: AUROC {parent_log.get('cv_mean_heldout_clip_auroc', float('nan')):.3f}"
            f" +/- {parent_log.get('cv_std_heldout_clip_auroc', 0):.3f}  "
            f"F1 {parent_log.get('cv_mean_heldout_clip_f1', float('nan')):.3f}"
            f" +/- {parent_log.get('cv_std_heldout_clip_f1', 0):.3f}"
        )
        if not dry_run and parent_log:
            with mlflow.start_run(run_id=parent.info.run_id):
                mlflow.log_metrics(parent_log)

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="ulcer_detection", help="MLflow experiment name")
    p.add_argument("--mlflow-uri", default=cfg.mlflow_db)
    p.add_argument(
        "--manifest",
        default=str(cfg.ulcer.heldout / "heldout_temporal_manifest.csv"),
        help="Held-out temporal manifest CSV",
    )
    p.add_argument("--dry-run", action="store_true", help="Compute but do not log to MLflow")
    p.add_argument("--force", action="store_true", help="Re-log even if already present")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    if args.dry_run:
        print("[DRY RUN] metrics will be computed but not logged.")
    process_experiment(args.experiment, args.manifest, dry_run=args.dry_run, force=args.force)
