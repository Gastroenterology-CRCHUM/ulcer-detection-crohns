"""Shared pytest fixtures for the test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def rgb_image() -> Image.Image:
    """Return a small square RGB image."""
    return Image.new("RGB", (256, 256), color=(128, 128, 128))


@pytest.fixture
def non_square_rgb_image() -> Image.Image:
    """Return a non-square RGB image for padding tests."""
    return Image.new("RGB", (50, 100), color=(255, 255, 255))


@pytest.fixture
def sample_dataset_df() -> pd.DataFrame:
    """Return a small manifest-style DataFrame."""
    return pd.DataFrame(
        {
            "relative_path": ["img_001.jpg", "img_002.jpg", "img_003.jpg"],
            "label": [0, 1, 0],
            "video_id": ["v001", "v002", "v003"],
            "segment_id": [1, 1, 2],
            "patient_id": ["p001", "p002", "p001"],
            "split": ["train", "train", "val"],
        }
    )


@pytest.fixture
def sample_dataset(tmp_path, sample_dataset_df):
    """Create a temporary dataset directory with images and a manifest DataFrame."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    for image_name in sample_dataset_df["relative_path"]:
        Image.new("RGB", (256, 256), color=(128, 128, 128)).save(data_dir / image_name)

    return data_dir, sample_dataset_df.copy()


@pytest.fixture
def sample_binary_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic probability/label pair."""
    np.random.seed(42)
    probabilities = np.random.rand(100).astype(np.float32)
    labels = np.random.randint(0, 2, 100).astype(int)
    return probabilities, labels


@pytest.fixture
def sample_clip_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return small clip-level aggregation inputs."""
    probabilities = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.7], dtype=float)
    predictions = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    clip_ids = ["clip_1", "clip_1", "clip_2", "clip_2", "clip_3", "clip_3"]
    return probabilities, predictions, labels, clip_ids


@pytest.fixture
def sample_threshold_results() -> list[dict]:
    """Return a tiny ordered list of threshold results."""
    return [
        {"threshold": 0.3, "f1": 0.70, "accuracy": 0.75, "precision": 0.80, "recall": 0.65},
        {"threshold": 0.4, "f1": 0.75, "accuracy": 0.80, "precision": 0.82, "recall": 0.70},
        {"threshold": 0.5, "f1": 0.73, "accuracy": 0.78, "precision": 0.79, "recall": 0.68},
        {"threshold": 0.6, "f1": 0.68, "accuracy": 0.72, "precision": 0.75, "recall": 0.62},
    ]


@pytest.fixture
def checkpoint_workspace(tmp_path) -> Path:
    """Return a temporary workspace root for checkpoint loading tests."""
    return tmp_path
