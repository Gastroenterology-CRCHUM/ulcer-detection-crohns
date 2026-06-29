"""
Post-hoc computation of clip-level heldout metrics for all CV runs.

For each parent run in the experiment, loads per-fold heldout frame predictions
from child runs, aggregates by clip (mean probability), applies the per-fold
clip threshold, and logs the resulting metrics back to MLflow:

  Child run  → heldout_clip__{metric}_mean   (per-fold clip metrics)
  Parent run → cv_mean_heldout_clip_{metric}  (mean across folds)
               cv_std_heldout_clip_{metric}   (std  across folds)

Usage
-----
    python scripts/ulcer/log_heldout_clip_metrics.py
    python scripts/ulcer/log_heldout_clip_metrics.py --experiment ulcer_detection --dry-run
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.paths import get_default_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_metrics(
    probs: np.ndarray,
    clip_keys: np.ndarray,
    clip_label_map: dict[str, int],
    threshold: float,
) -> dict[str, float]:
    """Aggregate frame probs by clip and compute classification metrics."""
    df = pd.DataFrame({"clip_key": clip_keys, "prob": probs})
    clip_df = df.groupby("clip_key", sort=False)["prob"].mean().reset_index()
    clip_df["label"] = clip_df["clip_key"].map(clip_label_map)
    clip_df = clip_df.dropna(subset=["label"])
    clip_df["label"] = clip_df["label"].astype(int)

    y = clip_df["label"].values
    p = clip_df["prob"].values
    preds = (p >= threshold).astype(int)

    tn = int(((y == 0) & (preds == 0)).sum())
    fp = int(((y == 0) & (preds == 1)).sum())
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auroc":       roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        "accuracy":    accuracy_score(y, preds),
        "f1":          f1_score(y, preds, zero_division=0),
        "precision":   precision_score(y, preds, zero_division=0),
        "sensitivity": recall_score(y, preds, zero_division=0),
        "specificity": spec,
        "n_clips":     len(clip_df),
    }


def _download_npy(client: MlflowClient, run_id: str, artifact_path: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, artifact_path, tmp)
        return np.load(local)


def _find_heldout_probs_path(client: MlflowClient, run_id: str) -> str | None:
    """Return the artifact path of the per-fold heldout probs npy."""
    try:
        arts = client.list_artifacts(run_id, "predictions/heldout")
        for a in arts:
            if "probs" in a.path:
                return a.path
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_experiment(
    experiment_name: str,
    manifest_path: str,
    dry_run: bool = False,
) -> None:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # Load heldout manifest once
    df_hm = pd.read_csv(manifest_path)
    clip_keys = df_hm["clip_key"].values
    clip_label_map: dict[str, int] = (
        df_hm.groupby("clip_key")["label"]
        .apply(lambda x: int(x.mode()[0]))
        .to_dict()
    )

    # Fetch all runs
    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=1000,
    )
    parents = [r for r in all_runs if not r.data.tags.get("mlflow.parentRunId")]
    print(f"Found {len(parents)} parent run(s) in '{experiment_name}'.")

    for parent in parents:
        model_name = parent.data.tags.get("model", parent.info.run_id[:8])
        parent_id  = parent.info.run_id
        children   = [r for r in all_runs
                      if r.data.tags.get("mlflow.parentRunId") == parent_id]

        if not children:
            print(f"  {model_name}: no child runs — skip.")
            continue

        # Check if already logged
        existing = {k for k in parent.data.metrics if "heldout_clip" in k}
        if existing:
            print(f"  {model_name}: clip heldout metrics already present — skip "
                  f"(use --force to overwrite).")
            continue

        print(f"  {model_name}: processing {len(children)} fold(s)...")
        fold_metrics: dict[str, list[float]] = {
            k: [] for k in ("auroc", "accuracy", "f1", "precision", "sensitivity", "specificity")
        }

        for ch in children:
            fold_name = ch.data.tags.get("mlflow.runName", ch.info.run_id[:8])
            clip_thresh = ch.data.metrics.get("fold_optimal_clip_threshold", 0.5)
            probs_path = _find_heldout_probs_path(client, ch.info.run_id)
            if probs_path is None:
                print(f"    {fold_name}: no heldout probs artifact — skip.")
                continue

            probs = _download_npy(client, ch.info.run_id, probs_path)
            m = _clip_metrics(probs, clip_keys, clip_label_map, clip_thresh)

            print(f"    {fold_name}: AUROC={m['auroc']:.3f}  F1={m['f1']:.3f}  "
                  f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
                  f"n_clips={m['n_clips']}")

            if not dry_run:
                with mlflow.start_run(run_id=ch.info.run_id):
                    mlflow.log_metrics({
                        f"heldout_clip__{k}_mean": v
                        for k, v in m.items()
                        if k != "n_clips" and not np.isnan(v)
                    })
                    mlflow.log_metric("heldout_clip_n_clips", m["n_clips"])

            for k in fold_metrics:
                if not np.isnan(m[k]):
                    fold_metrics[k].append(m[k])

        # Aggregate across folds and log to parent
        parent_log: dict[str, float] = {}
        for k, vals in fold_metrics.items():
            if not vals:
                continue
            arr = np.array(vals)
            parent_log[f"cv_mean_heldout_clip_{k}"] = float(arr.mean())
            parent_log[f"cv_std_heldout_clip_{k}"]  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        print(f"    → parent: AUROC {parent_log.get('cv_mean_heldout_clip_auroc', float('nan')):.3f} "
              f"± {parent_log.get('cv_std_heldout_clip_auroc', 0):.3f}  "
              f"F1 {parent_log.get('cv_mean_heldout_clip_f1', float('nan')):.3f} "
              f"± {parent_log.get('cv_std_heldout_clip_f1', 0):.3f}")

        if not dry_run and parent_log:
            with mlflow.start_run(run_id=parent_id):
                mlflow.log_metrics(parent_log)

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="ulcer_detection",
                   help="MLflow experiment name (default: ulcer_detection)")
    p.add_argument("--manifest",
                   default=str(cfg.ulcer.splits / "heldout_temporal_manifest.csv"),
                   help="Path to heldout temporal manifest CSV")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute metrics but do not log to MLflow")
    p.add_argument("--force", action="store_true",
                   help="Re-log even if clip heldout metrics already present")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.dry_run:
        print("[DRY RUN] metrics will be computed but not logged.")
    process_experiment(args.experiment, args.manifest, dry_run=args.dry_run)
