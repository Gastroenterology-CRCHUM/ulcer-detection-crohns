"""
scripts/ulcer/evaluate_with_delong.py
--------------------------------------
Multi-model evaluation with pairwise DeLong AUROC comparison.

Usage:
    python scripts/ulcer/evaluate_with_delong.py \\
        --run-id <MLflow CV parent run ID> \\
        --manifest <path to test CSV> \\
        --data-dir <path to image directory>

Outputs (in OUTPUT_DIR):
    roc_curves.png
    delong_heatmap.png
    model_comparison.csv
    delong_comparisons.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.delong import delong_matrix
from src.evaluation.metrics import compute_clip_metrics, compute_metrics_with_ci
from src.evaluation.plots import plot_delong_heatmap, plot_roc_curves

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output/evaluation")
THRESHOLD = 0.5
METRIC_COLS = ["Accuracy", "AUROC", "Sensitivity", "Specificity", "F1"]
DELONG_ALPHA = 0.05
MODEL_NAME_MAP: dict[str, str] = {}  # populate to override display names


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _clean_model_name(raw_name: str) -> str:
    """Strip training-mode suffixes and apply MODEL_NAME_MAP overrides."""
    for suffix in ("-allBackbone", "-freezeBackbone", "-9Backbone"):
        raw_name = raw_name.replace(suffix, "")
    return MODEL_NAME_MAP.get(raw_name, raw_name)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (probs, labels, video_ids) after a single inference pass.

    Expects batches of (images, labels, video_ids).
    Handles HuggingFace wrappers that expose a .logits attribute.
    Assumes scalar output per image (num_classes=1, sigmoid activation).
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_video_ids: list[str] = []

    for images, labels, video_ids in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if hasattr(logits, "logits"):
            logits = logits.logits
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_video_ids.extend(video_ids)

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        all_video_ids,
    )


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------


def _evaluate_one(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    threshold: float = 0.5,
) -> dict:
    """Return a dict containing probs, labels, ROC data, and CI metrics."""
    probs, labels, video_ids = _run_inference(model, dataloader, device)
    preds = (probs >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(labels, probs)

    return {
        "probs": probs,
        "labels": labels,
        "video_ids": video_ids,
        "fpr": fpr,
        "tpr": tpr,
        "frame_metrics": compute_metrics_with_ci(labels, preds, probs),
        "clip_metrics": compute_clip_metrics(labels, probs, video_ids, threshold),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    model_probs: dict[str, np.ndarray],
    labels: np.ndarray,
    device: torch.device | str,
) -> None:
    """Run DeLong pairwise comparison and save results to OUTPUT_DIR.

    Args:
        model_probs: Mapping of display name to predicted probability array.
                     Example: {"fold_1": probs_array, "ensemble": ens_probs}.
        labels:      Ground-truth binary label array (same length as each probs array).
        device:      Torch device (used for any downstream GPU operations).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build ROC data from provided probability arrays
    roc_data: list[dict] = []
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        roc_data.append({"name": name, "fpr": fpr, "tpr": tpr, "auc": auc})

    # ROC curves
    plot_roc_curves(roc_data, save_path=OUTPUT_DIR / "roc_curves.png")

    # Summary metrics per model
    rows = []
    for name, probs in model_probs.items():
        preds = (probs >= THRESHOLD).astype(int)
        frame_metrics = compute_metrics_with_ci(labels, preds, probs)
        rows.append({"Model": name, **frame_metrics})

    def _keep_display_cols(df: pd.DataFrame) -> pd.DataFrame:
        return df[["Model"] + [c for c in METRIC_COLS if c in df.columns]]

    df_metrics = _keep_display_cols(pd.DataFrame(rows))
    df_metrics.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nFrame-level metrics — mean (95% CI)")
    print(df_metrics.to_string(index=False))

    # DeLong pairwise test
    print("\n" + "=" * 60)
    print("DELONG TEST — Pairwise AUROC comparisons")
    print("=" * 60)

    p_matrix, df_delong = delong_matrix(
        labels=labels,
        model_probs=model_probs,
        alpha=DELONG_ALPHA,
    )

    print(df_delong.to_string(index=False))
    df_delong.to_csv(OUTPUT_DIR / "delong_comparisons.csv", index=False)

    n_sig = (df_delong["significant"] == "✓").sum()
    print(f"\n{n_sig} / {len(df_delong)} pairs significantly different (alpha = {DELONG_ALPHA})")

    plot_delong_heatmap(
        p_matrix,
        df_delong,
        alpha=DELONG_ALPHA,
        save_path=OUTPUT_DIR / "delong_heatmap.png",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate CV ensemble with DeLong pairwise AUROC comparison"
    )
    parser.add_argument("--run-id", required=True, help="MLflow CV parent run ID")
    parser.add_argument("--manifest", required=True, help="Path to test manifest CSV")
    parser.add_argument("--data-dir", required=True, help="Path to image directory")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)"
    )
    args = parser.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    from src.config.loader import load_config
    from src.utils import get_device

    device = get_device()
    cfg = load_config()

    mlflow.set_tracking_uri(cfg.paths.mlflow_db)
    client = MlflowClient()
    parent_run = client.get_run(args.run_id)
    model_key = parent_run.data.tags.get("model", "")
    if not model_key:
        raise ValueError(f"Could not find 'model' tag on run {args.run_id}")

    # Collect child fold runs
    child_runs = client.search_runs(
        experiment_ids=[parent_run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{args.run_id}'",
    )
    if not child_runs:
        raise ValueError(f"No child fold runs found under parent run {args.run_id}")

    print(f"Found {len(child_runs)} fold runs for model '{model_key}'")

    # Load each fold's saved predictions/probabilities from MLflow artifacts
    # (assumes each fold run logged a test_probs.npy and test_labels.npy artifact)
    fold_probs: dict[str, np.ndarray] = {}
    labels_arr: np.ndarray | None = None

    import tempfile

    for run in child_runs:
        run_id = run.info.run_id
        fold_name = run.data.tags.get("fold", run_id[:8])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            try:
                client.download_artifacts(run_id, "test_probs.npy", tmp)
                client.download_artifacts(run_id, "test_labels.npy", tmp)
                probs = np.load(tmp_path / "test_probs.npy")
                lbl = np.load(tmp_path / "test_labels.npy")
                fold_probs[f"fold_{fold_name}"] = probs
                if labels_arr is None:
                    labels_arr = lbl
            except Exception as e:
                print(f"  Warning: could not load artifacts for fold {fold_name}: {e}")

    if not fold_probs or labels_arr is None:
        raise RuntimeError(
            "No fold probability artifacts could be loaded from MLflow. "
            "Ensure each fold run logged 'test_probs.npy' and 'test_labels.npy' artifacts."
        )

    # Ensemble = mean of fold probabilities
    stacked = np.stack(list(fold_probs.values()), axis=0)
    fold_probs["ensemble"] = stacked.mean(axis=0)

    main(
        model_probs=fold_probs,
        labels=labels_arr,
        device=device,
    )
