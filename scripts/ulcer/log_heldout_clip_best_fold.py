"""
Post-hoc: compute clip-level heldout CI for the best-fold model.

Loads `predictions/heldout_best_fold_probs.npy`, aggregates by clip
(mean probability), finds the best-fold clip threshold from the
corresponding child run, then computes bootstrap CI and logs:

  Artifact  → metrics/heldout_clip_ci.json
  Metrics   → heldout_clip__{metric}_mean / _lower / _upper

Usage
-----
    python scripts/ulcer/log_heldout_clip_best_fold.py
    python scripts/ulcer/log_heldout_clip_best_fold.py --dry-run
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

import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient

from src.config.paths import get_default_paths
from src.evaluation.metrics import compute_metrics_with_ci


def _download_npy(client: MlflowClient, run_id: str, path: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, path, tmp)
        return np.load(local)


def _get_best_fold_clip_threshold(
    client: MlflowClient,
    exp_id: str,
    parent_id: str,
) -> float:
    """Return clip threshold of the best val-AUROC fold."""
    children = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        max_results=20,
    )
    if children.empty:
        return 0.5
    best_row = children.loc[
        children["metrics.fold_val_auroc"].fillna(-1).idxmax()
    ]
    return float(best_row.get("metrics.fold_optimal_clip_threshold", 0.5))


def process_experiment(
    experiment_name: str,
    manifest_path: str,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    df_hm = pd.read_csv(manifest_path)
    clip_keys = df_hm["clip_key"].values
    df_clip_ref = (
        df_hm.groupby("clip_key", sort=False)
        .agg(clip_label=("label", lambda x: int(x.mode()[0])))
        .reset_index()
    )

    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id], max_results=1000
    )
    parents = [r for r in all_runs if not r.data.tags.get("mlflow.parentRunId")]
    print(f"Found {len(parents)} parent run(s).")

    for parent in parents:
        model = parent.data.tags.get("model", parent.info.run_id[:8])
        run_id = parent.info.run_id

        # Skip if already logged
        if not force:
            arts = [a.path for a in client.list_artifacts(run_id, "metrics")]
            if "metrics/heldout_clip_ci.json" in arts:
                print(f"  {model}: heldout_clip_ci.json already present — skip.")
                continue

        # Load best-fold heldout frame probs
        try:
            probs = _download_npy(client, run_id, "predictions/heldout_best_fold_probs.npy")
        except Exception as e:
            print(f"  {model}: cannot load heldout_best_fold_probs — {e}")
            continue

        # Aggregate by clip
        df_tmp = pd.DataFrame({"clip_key": clip_keys, "prob": probs})
        clip_df = (
            df_tmp.groupby("clip_key", sort=False)["prob"]
            .mean()
            .reset_index()
            .merge(df_clip_ref, on="clip_key")
        )
        clip_probs  = clip_df["prob"].values
        clip_labels = clip_df["clip_label"].values

        # Get clip threshold from best fold child run
        clip_thresh = _get_best_fold_clip_threshold(
            client, exp.experiment_id, run_id
        )
        clip_preds = (clip_probs >= clip_thresh).astype(int)

        print(f"  {model}: {len(clip_df)} clips, thresh={clip_thresh:.3f}")

        # Bootstrap CI
        metrics = compute_metrics_with_ci(
            clip_labels, clip_preds, clip_probs,
            n_bootstrap=1000, seed=42,
        )

        # Build CI dict (same format as heldout_ci.json)
        ci: dict[str, dict] = {}
        for k, v in metrics.items():
            if not (k.startswith("_") and isinstance(v, float) and not np.isnan(v)):
                continue
            if k.endswith("_lower"):
                name = k[1:-6]
                ci.setdefault(name, {})["lower"] = round(float(v), 4)
            elif k.endswith("_upper"):
                name = k[1:-6]
                ci.setdefault(name, {})["upper"] = round(float(v), 4)
            elif k.endswith("_mean"):
                name = k[1:-5]
                ci.setdefault(name, {})["mean"] = round(float(v), 4)

        for name, d in ci.items():
            print(f"    {name}: {d.get('mean','?'):.3f} "
                  f"({d.get('lower','?'):.3f}–{d.get('upper','?'):.3f})")

        if dry_run:
            continue

        with mlflow.start_run(run_id=run_id):
            # Log CI artifact
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "heldout_clip_ci.json")
                with open(path, "w") as f:
                    json.dump(ci, f, indent=2)
                mlflow.log_artifact(path, artifact_path="metrics")

            # Log scalar metrics
            mlflow.log_metrics({
                f"heldout_clip__{name}_mean": d["mean"]
                for name, d in ci.items() if "mean" in d
            })
            mlflow.log_metric("heldout_clip_n_clips", len(clip_df))

        print(f"    → logged heldout_clip_ci.json")

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="ulcer_detection")
    p.add_argument("--manifest",
                   default=str(cfg.ulcer.splits / "heldout_temporal_manifest.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-compute even if heldout_clip_ci.json already present")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.dry_run:
        print("[DRY RUN]")
    process_experiment(args.experiment, args.manifest,
                       dry_run=args.dry_run, force=args.force)
