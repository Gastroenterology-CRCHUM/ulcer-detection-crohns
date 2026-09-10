"""
src/evaluation/mlflow_io.py
-----------------------------
One tiny MLflow artifact-download helper, shared by the reporting/backfill
scripts that read per-fold .npy prediction artifacts. Deliberately NOT in
src/evaluation/mlflow_utils.py, which imports torch and mlflow.pytorch at
module scope -- every caller of this module is a read-only reporting/backfill
script with no reason to load torch.

Public API
----------
download_npy(client, run_id, artifact_path) -> np.ndarray
    Download one artifact to a temp dir and load it with np.load.
"""

from __future__ import annotations

import tempfile

import numpy as np
from mlflow import MlflowClient


def download_npy(client: MlflowClient, run_id: str, artifact_path: str) -> np.ndarray:
    """Download one .npy artifact from an MLflow run and load it."""
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, artifact_path, tmp)
        return np.load(local)
