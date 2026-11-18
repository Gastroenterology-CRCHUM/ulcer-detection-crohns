"""
src/evaluation/mlflow_io.py
---------------------------
Read side of MLflow for the result scripts: finding the CV runs and loading
their artifacts. Writing during training lives in mlflow_utils.py, which
imports torch at module scope, so it is deliberately kept separate.

Public API
----------
load_cv_runs(client, experiment_name) -> {model_key: CVRuns(parent, children)}
    One CV parent run per MODEL_REGISTRY key, children sorted by fold.

artifact_paths(client, run_id, folder) -> list[str]
download_npy(client, run_id, artifact_path) -> np.ndarray
"""

from __future__ import annotations

import tempfile
from typing import NamedTuple

import numpy as np
from mlflow import MlflowClient
from mlflow.entities import Run

from src.config.models import MODEL_REGISTRY


class CVRuns(NamedTuple):
    parent: Run
    children: list[Run]


def load_cv_runs(client: MlflowClient, experiment_name: str) -> dict[str, CVRuns]:
    """CV parent run (training_mode "cv_*") and its fold runs, per registry model.

    Models are returned in MODEL_REGISTRY order; a model without a parent or
    without fold runs is skipped with a warning. If a model has several CV
    parents (re-trained), the most recent one is used and the others are named.
    """
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found.")
    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)

    parents: dict[str, list[Run]] = {}
    children: dict[str, list[Run]] = {}
    for run in runs:
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if parent_id:
            children.setdefault(parent_id, []).append(run)
        elif run.data.tags.get("training_mode", "").startswith("cv_"):
            parents.setdefault(run.data.tags.get("model"), []).append(run)

    out: dict[str, CVRuns] = {}
    for model_key in MODEL_REGISTRY:
        candidates = sorted(parents.get(model_key, []), key=lambda r: r.info.start_time)
        if not candidates:
            print(f"  [warn] {model_key}: no CV parent run found, skipping.")
            continue
        parent = candidates[-1]
        if len(candidates) > 1:
            older = ", ".join(r.info.run_id[:8] for r in candidates[:-1])
            print(
                f"  [warn] {model_key}: {len(candidates)} CV parent runs, using the latest "
                f"({parent.info.run_id[:8]}), ignoring {older}."
            )
        folds = sorted(
            children.get(parent.info.run_id, []), key=lambda r: int(r.data.params.get("fold", -1))
        )
        if not folds:
            print(f"  [warn] {model_key}: no child fold runs, skipping.")
            continue
        out[model_key] = CVRuns(parent, folds)
    return out


def artifact_paths(client: MlflowClient, run_id: str, folder: str) -> list[str]:
    """Artifact paths directly under `folder`, or [] when the folder is missing."""
    try:
        return [a.path for a in client.list_artifacts(run_id, folder)]
    except Exception:
        return []


def download_npy(client: MlflowClient, run_id: str, artifact_path: str) -> np.ndarray:
    """Download one .npy artifact from an MLflow run and load it."""
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, artifact_path, tmp)
        return np.load(local)
