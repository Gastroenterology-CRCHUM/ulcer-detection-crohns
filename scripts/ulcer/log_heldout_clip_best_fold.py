"""
scripts/ulcer/log_heldout_clip_best_fold.py
-------------------------------------------
Post-hoc clip-level held-out CI for the best-fold model, logged back to MLflow.

Loads the parent run's `predictions/heldout_best_fold_probs.npy`, averages it
per clip, applies the clip threshold of the fold with the best validation AUROC,
computes row-wise bootstrap CIs (compute_metrics_with_ci, as during training)
and logs:

  Artifact -> metrics/heldout_clip_ci.json
  Metrics  -> heldout_clip__{metric}_mean, heldout_clip_n_clips

All held-out frames are used (no informative-frame filter), as in the values
already logged.

Usage
-----
    python -m scripts.ulcer.log_heldout_clip_best_fold
    python -m scripts.ulcer.log_heldout_clip_best_fold --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.config.paths import get_default_paths  # noqa: E402
from src.evaluation.heldout import aggregate_to_clip, load_cohort  # noqa: E402
from src.evaluation.metrics import compute_metrics_with_ci  # noqa: E402
from src.evaluation.mlflow_io import artifact_paths, download_npy, load_cv_runs  # noqa: E402


def _best_fold_clip_threshold(children) -> float:
    """Clip threshold of the fold with the best validation AUROC (0.5 if none)."""
    if not children:
        return 0.5
    best = max(children, key=lambda r: r.data.metrics.get("fold_val_auroc", -1))
    return float(best.data.metrics.get("fold_optimal_clip_threshold", 0.5))


def _ci_summary(metrics: dict) -> dict[str, dict]:
    """{metric: {mean, lower, upper}} from compute_metrics_with_ci's "_<Metric>_<stat>" keys,
    the format of the training-time heldout_ci.json."""
    ci: dict[str, dict] = {}
    for k, v in metrics.items():
        if not (k.startswith("_") and isinstance(v, float) and not np.isnan(v)):
            continue
        for stat in ("lower", "upper", "mean"):
            if k.endswith(f"_{stat}"):
                ci.setdefault(k[1 : -len(stat) - 1], {})[stat] = round(float(v), 4)
    return ci


def process_experiment(
    experiment_name: str, manifest_path: str, dry_run: bool = False, force: bool = False
) -> None:
    client = MlflowClient()
    cohort = load_cohort(manifest_path)
    # First-appearance clip order: the row-wise bootstrap resamples by position, so the
    # order is kept as in the values already logged.
    clip_order = list(pd.unique(cohort.frame_clip_keys))
    clip_labels = cohort.clip_labels_in(clip_order)

    runs = load_cv_runs(client, experiment_name)
    print(f"Found {len(runs)} CV parent run(s).")

    for model_key, (parent, children) in runs.items():
        run_id = parent.info.run_id
        if not force and "metrics/heldout_clip_ci.json" in artifact_paths(
            client, run_id, "metrics"
        ):
            print(f"  {model_key}: heldout_clip_ci.json already present, skip.")
            continue

        try:
            probs = download_npy(client, run_id, "predictions/heldout_best_fold_probs.npy")
        except Exception as e:
            print(f"  {model_key}: cannot load heldout_best_fold_probs, {e}")
            continue

        clip_probs = aggregate_to_clip(probs, cohort.frame_clip_keys, clip_order)
        threshold = _best_fold_clip_threshold(children)
        print(f"  {model_key}: {len(clip_order)} clips, thresh={threshold:.3f}")

        metrics = compute_metrics_with_ci(
            clip_labels,
            (clip_probs >= threshold).astype(int),
            clip_probs,
            n_bootstrap=1000,
            seed=42,
        )
        ci = _ci_summary(metrics)
        for name, d in ci.items():
            print(
                f"    {name}: {d.get('mean', '?'):.3f} "
                f"({d.get('lower', '?'):.3f}–{d.get('upper', '?'):.3f})"
            )

        if dry_run:
            continue

        with mlflow.start_run(run_id=run_id):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "heldout_clip_ci.json")
                with open(path, "w") as f:
                    json.dump(ci, f, indent=2)
                mlflow.log_artifact(path, artifact_path="metrics")
            mlflow.log_metrics(
                {f"heldout_clip__{name}_mean": d["mean"] for name, d in ci.items() if "mean" in d}
            )
            mlflow.log_metric("heldout_clip_n_clips", len(clip_order))
        print("    -> logged heldout_clip_ci.json")

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", default="ulcer_detection")
    p.add_argument("--mlflow-uri", default=cfg.mlflow_db)
    p.add_argument("--manifest", default=str(cfg.ulcer.heldout / "heldout_temporal_manifest.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-compute even if heldout_clip_ci.json already present",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    if args.dry_run:
        print("[DRY RUN]")
    process_experiment(args.experiment, args.manifest, dry_run=args.dry_run, force=args.force)
