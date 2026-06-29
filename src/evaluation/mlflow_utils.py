"""
src/evaluation/mlflow_utils.py
==============================
MLflow utilities — model registry, artifact logging, run tagging.

Public API
----------
set_run_tags             — set standard tags on the current run
log_dataset_info         — log manifest statistics as MLflow params
log_figures              — log a {name: Figure} dict as PNG artifacts
log_confusion_matrix     — log the confusion matrix as an image
log_split_metrics        — log frame + clip metrics with a split prefix
register_best_model      — register a run in the Model Registry
promote_model            — assign the "champion" alias to a model version
get_champion             — return the URI of the champion model
get_best_run             — find the run with the best metric value
compare_runs_to_markdown — generate a Markdown table comparing runs
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.evaluation.plots import plot_confusion_matrix

# ---------------------------------------------------------------------------
# Tags standards
# ---------------------------------------------------------------------------


def set_run_tags(
    model_name: str,
    mode: str,
    extra: dict | None = None,
) -> None:
    """Set standard tags (model, mode, GPU, host, git) on the current run."""
    tags: dict[str, str] = {
        "model": model_name,
        "training_mode": mode,
        "host": socket.gethostname(),
        "platform": platform.system(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        tags["git_commit"] = sha
    except Exception:
        pass

    if extra:
        tags.update({str(k): str(v) for k, v in extra.items()})
    mlflow.set_tags(tags)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def log_figures(figures: dict[str, plt.Figure], subdir: str = "") -> None:
    """
    Log a {filename: Figure} dict as PNG artifacts in MLflow.

    Args:
        figures : {stem: fig} — ".png" is appended automatically.
        subdir  : subdirectory inside the artifacts (e.g. "test").
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, fig in figures.items():
            tmp_path = Path(tmpdir) / f"{name}.png"
            try:
                fig.savefig(tmp_path, format="png", dpi=150, bbox_inches="tight")
                mlflow.log_artifact(str(tmp_path), artifact_path=subdir or None)
            finally:
                plt.close(fig)


def log_figures_from_dir(fig_dir: Path, subdir: str = "") -> None:
    """
    Log all files in a directory as PNG artifacts in MLflow.

    Args:
        fig_dir : directory containing the images to log.
        subdir  : subdirectory inside the artifacts (e.g. "test").
    """
    for img_path in fig_dir.glob("*.png"):
        mlflow.log_artifact(str(img_path), artifact_path=subdir or None)


def log_confusion_matrix(cm: np.ndarray, threshold: float, model_name: str = "", prefix: str | None = None) -> None:
    """Log the confusion matrix as a PNG image."""
    fig = plot_confusion_matrix(cm, threshold, title=f"Confusion matrix - Ulcer detection ({model_name})")
    key = f"{model_name}_confusion_matrix"
    log_figures({key: fig}, subdir=prefix)


def log_split_metrics(metrics: dict, split: str, step: int | None = None) -> None:
    """
    Log metrics for a split from the output of compute_metrics_with_ci.

    compute_metrics_with_ci returns keys of the form "_F1_mean",
    "_AUROC_mean", etc. (underscore prefix, float value).
    These are transformed into "test__f1_mean", "test__auroc_mean", etc.
    The double underscore (split + "_") preserves readability in the MLflow UI.

    Args:
        metrics : output of compute_metrics_with_ci.
        split   : "train" | "val" | "test".
        step    : epoch or fold (optional).
    """
    # Only _mean values → mlflow.log_metrics (time-series capable, appear in Metrics tab).
    # CI bounds (_lower/_upper) are stored in a JSON artifact via log_ci_artifact —
    # they are not time-series data and must not generate spurious metric plots.
    means = {
        f"{split}__{k[1:].lower()}": v
        for k, v in metrics.items()
        if k.startswith("_") and k.endswith("_mean") and isinstance(v, float)
    }
    if means:
        mlflow.log_metrics(means, step=step)


def log_ci_artifact(metrics: dict, split: str) -> None:
    """Write bootstrap CI bounds to a JSON artifact in MLflow.

    CI bounds are stored as an artifact rather than MLflow metrics to avoid
    cluttering the Metrics tab with non-time-series values.

    Artifact path: ``metrics/<split>_ci.json``

    Args:
        metrics : output of compute_metrics_with_ci (contains _lower/_upper keys).
        split   : "test" | "test_clip" | etc.
    """
    ci: dict[str, dict[str, float]] = {}
    for k, v in metrics.items():
        if not (k.startswith("_") and isinstance(v, float) and not np.isnan(v)):
            continue
        if k.endswith("_lower"):
            name = k[1:-6]  # strip leading _ and trailing _lower
            ci.setdefault(name, {})["lower"] = round(float(v), 4)
        elif k.endswith("_upper"):
            name = k[1:-6]
            ci.setdefault(name, {})["upper"] = round(float(v), 4)
        elif k.endswith("_mean"):
            name = k[1:-5]
            ci.setdefault(name, {})["mean"] = round(float(v), 4)

    if not ci:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{split}_ci.json"
        path.write_text(json.dumps(ci, indent=2))
        mlflow.log_artifact(str(path), artifact_path="metrics")


def log_dataset_info(manifest_path: Path) -> None:
    """Log manifest statistics as MLflow parameters."""
    df = pd.read_csv(manifest_path)

    for split in ("train", "val", "test"):
        sdf = df[df["split"] == split]
        n = len(sdf)
        if n == 0:
            continue
        n_pos = (sdf["label"] == 1).sum()
        mlflow.log_params(
            {
                f"data_{split}_n": n,
                f"data_{split}_n_positive": int(n_pos),
                f"data_{split}_pos_ratio": round(n_pos / n, 4),
            }
        )

    if "patient_id" in df.columns:
        mlflow.log_param("data_n_patients", df["patient_id"].nunique())
    if "video_id" in df.columns:
        mlflow.log_param("data_n_videos", df["video_id"].nunique())
    mlflow.log_param("data_total_frames", len(df))


def log_size_distribution(manifest_path: Path, label_col: str) -> None:
    """Log the ulcer size distribution as MLflow parameters."""
    df = pd.read_csv(manifest_path)
    if label_col not in df.columns:
        return

    size_counts = df[label_col].value_counts().sort_index()
    for size, count in size_counts.items():
        mlflow.log_param(f"size_{size}_count", int(count))
    mlflow.log_param("size_total", int(size_counts.sum()))


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


def register_best_model(
    run_id: str,
    model_name: str,
    metric: str = "test__f1_mean",
    artifact_path: str = "model",
    description: str = "",
) -> str | None:
    """Register a run's model in the MLflow Model Registry."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        try:
            client.create_registered_model(
                model_name,
                description=f"Ulcer detection — {model_name}",
            )
            print(f"  [Registry] Registered model created: '{model_name}'")
        except MlflowException:
            pass  # already exists

        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=description or f"Registered from run {run_id[:8]}",
        )
        print(f"  [Registry] '{model_name}' version {mv.version}  (run {run_id[:8]})")
        return mv.version
    except Exception as exc:
        print(f"  [Registry] Error: {exc}")
        return None


def promote_model(model_name: str, version: str | int, alias: str = "champion") -> None:
    """Set the *alias* on a Registry model version."""
    client = MlflowClient()
    try:
        client.set_registered_model_alias(model_name, alias, str(version))
        print(f"  [Registry] '{model_name}' v{version} → alias '{alias}'")
    except Exception as exc:
        print(f"  [Registry] Failed to promote: {exc}")


def get_champion(model_name: str) -> str:
    """Return the URI of the champion model for direct loading."""
    return f"models:/{model_name}@champion"


# ---------------------------------------------------------------------------
# Run comparison helpers
# ---------------------------------------------------------------------------


def get_best_run(
    experiment_name: str,
    metric: str,
    higher_is_better: bool = True,
) -> dict | None:
    """Find the run with the best value of *metric*."""
    client = MlflowClient()
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None
        order = "DESC" if higher_is_better else "ASC"
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"metrics.{metric} > 0",
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        if not runs:
            return None
        r = runs[0]
        return {
            "run_id": r.info.run_id,
            "metrics": r.data.metrics,
            "params": r.data.params,
            "tags": r.data.tags,
        }
    except Exception as exc:
        print(f"  Error in get_best_run: {exc}")
        return None


def compare_runs_to_markdown(
    experiment_name: str,
    metrics: list[str],
    n_runs: int = 10,
    order_by_metric: str | None = None,
    save_path: Path | None = None,
) -> str:
    """
    Generate a Markdown table of the top-N top-level runs.

    Nested runs (CV folds, Optuna trials) are excluded.
    MLflow sets the ``mlflow.parentRunId`` tag on child runs;
    top-level runs do NOT have this tag (absent != empty string).
    Filtering is therefore done client-side after fetching, not with a
    server-side filter_string that would compare against '' (empty string,
    not absent).
    """
    client = MlflowClient()
    order_by = order_by_metric or metrics[0]
    missing = "N/A"

    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return f"Experiment '{experiment_name}' not found."

        # Fetch more runs than needed to absorb nested runs,
        # then filter client-side on the absence of the parentRunId tag.
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{order_by} DESC"],
            max_results=n_runs * 5,
        )
    except Exception as exc:
        return f"Error: {exc}"

    # Top-level runs = those that do NOT have the mlflow.parentRunId tag
    top_level = [r for r in runs if not r.data.tags.get("mlflow.parentRunId")][:n_runs]

    if not top_level:
        return "No top-level runs found."

    rows = []
    for r in top_level:
        row = {
            "run": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
            "model": r.data.params.get("model", missing),
        }
        for m in metrics:
            v = r.data.metrics.get(m)
            row[m] = f"{v:.4f}" if v is not None else missing
        rows.append(row)

    df = pd.DataFrame(rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)
    )
    md = f"# {experiment_name}\n\n{header}\n{sep}\n{body}\n"

    if save_path:
        Path(save_path).write_text(md, encoding="utf-8")

    return md
