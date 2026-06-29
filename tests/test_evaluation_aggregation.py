"""
tests/test_evaluation_aggregation.py
====================================
Unit tests for clip-level aggregation helpers.
"""

import numpy as np
import pytest

from src.evaluation.aggregation import aggregate_frame_to_clip, compare_aggregation_methods


class TestAggregateFrameToClip:
    """Test aggregate_frame_to_clip function."""

    def test_mean_prob_method(self, sample_clip_arrays):
        probabilities, predictions, labels, clip_ids = sample_clip_arrays
        result = aggregate_frame_to_clip(
            probabilities, predictions, labels, clip_ids, method="mean_prob-0.5"
        )

        assert result["method"] == "mean_prob-0.5"
        assert result["n_clips"] == 3
        assert result["n_frames"] == 6
        assert list(result["clip_ids"]) == ["clip_1", "clip_2", "clip_3"]
        assert result["y_true"].shape == (3,)
        assert result["y_pred"].shape == (3,)
        assert result["y_prob_clip"].shape == (3,)

    def test_majority_vote_method(self, sample_clip_arrays):
        probabilities, predictions, labels, clip_ids = sample_clip_arrays
        result = aggregate_frame_to_clip(
            probabilities, predictions, labels, clip_ids, method="majority_vote"
        )

        assert result["method"] == "majority_vote"
        assert result["accuracy"] >= 0.0
        assert result["f1"] >= 0.0

    def test_threshold_ratio_method(self, sample_clip_arrays):
        probabilities, predictions, labels, clip_ids = sample_clip_arrays
        result = aggregate_frame_to_clip(
            probabilities, predictions, labels, clip_ids, method="threshold_ratio-0.5"
        )

        assert result["method"] == "threshold_ratio-0.5"
        assert result["confusion_matrix"].shape in ((2, 2), (1, 1))

    def test_invalid_method_raises(self, sample_clip_arrays):
        probabilities, predictions, labels, clip_ids = sample_clip_arrays
        with pytest.raises(ValueError):
            aggregate_frame_to_clip(
                probabilities, predictions, labels, clip_ids, method="unknown-0.5"
            )


class TestCompareAggregationMethods:
    """Test compare_aggregation_methods function."""

    def test_returns_ranked_dataframe(self):
        probabilities = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.7], dtype=float)
        predictions = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        labels = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        clip_ids = ["clip_1", "clip_1", "clip_2", "clip_2", "clip_3", "clip_3"]

        result = compare_aggregation_methods(
            probabilities, predictions, labels, clip_ids, n_bootstrap=100
        )

        assert list(result.columns) == [
            "method",
            "f1",
            "f1_95ci",
            "auroc",
            "auroc_95ci",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "n_clips",
        ]
        assert len(result) == 22  # 1 + 3×5 thresholds + 2×3 topk = 22
        assert result.iloc[0]["f1"] >= result.iloc[-1]["f1"]

    def test_methods_are_unique(self):
        probabilities = np.array([0.1, 0.4, 0.8, 0.9, 0.2, 0.7], dtype=float)
        predictions = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        labels = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        clip_ids = ["clip_1", "clip_1", "clip_2", "clip_2", "clip_3", "clip_3"]

        result = compare_aggregation_methods(
            probabilities, predictions, labels, clip_ids, n_bootstrap=50
        )
        assert result["method"].is_unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
