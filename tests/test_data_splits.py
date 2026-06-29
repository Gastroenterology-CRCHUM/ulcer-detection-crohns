"""
tests/test_data_splits.py
==========================
Unit tests for dataset splits and stratification.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    assign_cv_folds,
    assign_val_split,
    dominant_ulcer_size,
    patient_strat_label,
    patient_strat_labels,
)


class TestPatientStratLabel:
    """Test patient_strat_label functions."""

    def test_no_ulcer_patient(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p001", "p001", "p002"],
                "label": [0, 0, 1],
            }
        )
        assert patient_strat_label("p001", df) == "no_ulcer"

    def test_low_ulcer_patient(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p001", "p001", "p001", "p001", "p001"],
                "label": [1, 0, 0, 0, 0],
            }
        )
        assert patient_strat_label("p001", df) == "low_ulcer"

    def test_high_ulcer_patient(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p001", "p001", "p001", "p001", "p001"],
                "label": [1, 1, 1, 0, 1],
            }
        )
        assert patient_strat_label("p001", df) == "high_ulcer"

    def test_patient_strat_labels_vectorized(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p001", "p001", "p002", "p003", "p003"],
                "label": [1, 0, 0, 1, 1],
            }
        )
        labels = patient_strat_labels(df)
        assert len(labels) == len(df)
        assert labels[0] == labels[1]
        assert labels[3] == labels[4]


class TestDominantUlcerSize:
    """Test dominant_ulcer_size function."""

    def test_no_size_returns_none(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p001", "p001"],
                "label": [0, 0],
                "ulcer_size": [np.nan, np.nan],
            }
        )
        assert dominant_ulcer_size("p001", df) == "none"

    def test_mode_size(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p001", "p001", "p001", "p001"],
                "label": [1, 1, 1, 0],
                "ulcer_size": [0, 1, 1, 1],
            }
        )
        assert dominant_ulcer_size("p001", df) == "1"


class TestAssignValSplit:
    """Test assign_val_split function."""

    def test_train_val_split_ratio(self):
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i:03d}" for i in range(100)],
                "label": [0] * 70 + [1] * 30,
            }
        )
        train_df, val_df = assign_val_split(df, val_ratio=0.2, random_seed=42)
        assert len(train_df) + len(val_df) == len(df)
        assert 10 <= len(val_df) <= 30

    def test_split_no_overlap(self):
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i:03d}" for i in range(50)],
                "label": np.random.randint(0, 2, 50),
            }
        )
        train_df, val_df = assign_val_split(df, val_ratio=0.3, random_seed=42)
        assert set(train_df["patient_id"]).isdisjoint(set(val_df["patient_id"]))

    def test_split_preserves_columns(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p001", "p002", "p003"],
                "label": [0, 1, 0],
                "ulcer_size": [np.nan, 1, np.nan],
            }
        )
        train_df, val_df = assign_val_split(df, val_ratio=0.33, random_seed=42)
        assert set(train_df.columns) == set(df.columns)
        assert set(val_df.columns) == set(df.columns)


class TestAssignCVFolds:
    """Test assign_cv_folds function."""

    def test_cv_folds_created(self):
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i:03d}" for i in range(100)],
                "label": [i % 2 for i in range(100)],  # 50 each class, deterministic
            }
        )
        result_df = assign_cv_folds(df, n_splits=5, random_seed=42)
        assert "fold" in result_df.columns
        assert len(result_df["fold"].unique()) == 5

    def test_cv_no_patient_leakage(self):
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i:03d}" for i in range(50)],
                "label": [i % 2 for i in range(50)],  # 25 each class, deterministic
            }
        )
        result_df = assign_cv_folds(df, n_splits=5, random_seed=42)
        for patient_id in result_df["patient_id"].unique():
            assert len(result_df[result_df["patient_id"] == patient_id]["fold"].unique()) == 1

    def test_random_seed_reproducibility(self):
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i:03d}" for i in range(50)],
                "label": [i % 2 for i in range(50)],  # 25 each class, deterministic
            }
        )
        result1 = assign_cv_folds(df.copy(), n_splits=5, random_seed=42)
        result2 = assign_cv_folds(df.copy(), n_splits=5, random_seed=42)
        assert (result1["fold"] == result2["fold"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
