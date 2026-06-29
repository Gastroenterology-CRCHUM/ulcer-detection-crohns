"""
tests/test_evaluation_metrics.py
=================================
Unit tests for evaluation metrics.
"""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics_with_ci


class TestComputeMetricsWithCI:
    """Test compute_metrics_with_ci function."""

    @pytest.fixture
    def perfect_predictions(self):
        """Create perfect predictions."""
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        predictions = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        probs = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.9], dtype=np.float32)
        return labels, predictions, probs

    @pytest.fixture
    def random_predictions(self):
        """Create random predictions using a local RNG to avoid global state mutation."""
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, 100).astype(int)
        predictions = rng.integers(0, 2, 100).astype(int)
        probs = rng.random(100).astype(np.float32)
        return labels, predictions, probs

    def test_metrics_structure(self, perfect_predictions):
        """Test that metrics dict has required keys."""
        labels, predictions, probs = perfect_predictions
        metrics = compute_metrics_with_ci(labels, predictions, probs)

        assert isinstance(metrics, dict)
        required_keys = {
            "Accuracy",
            "AUROC",
            "Sensitivity",
            "Specificity",
            "F1",
            "_Accuracy_mean",
            "_AUROC_mean",
            "_Sensitivity_mean",
            "_Specificity_mean",
            "_F1_mean",
        }
        assert required_keys.issubset(metrics.keys())

    def test_perfect_predictions_metrics(self, perfect_predictions):
        """Test metrics for perfect predictions."""
        labels, predictions, probs = perfect_predictions
        metrics = compute_metrics_with_ci(labels, predictions, probs)

        assert metrics["_Accuracy_mean"] == 1.0
        assert metrics["_F1_mean"] == 1.0
        assert metrics["_Sensitivity_mean"] == 1.0

    def test_all_negative_class(self):
        """Test metrics when all samples are negative class."""
        labels = np.zeros(10, dtype=int)
        predictions = np.zeros(10, dtype=int)
        probs = np.full(10, 0.1, dtype=np.float32)

        metrics = compute_metrics_with_ci(labels, predictions, probs)

        assert metrics["_Accuracy_mean"] == 1.0
        # When there are no positive samples, precision and recall may be undefined
        # but should not cause errors

    def test_all_positive_class(self):
        """Test metrics when all samples are positive class."""
        labels = np.ones(10, dtype=int)
        predictions = np.ones(10, dtype=int)
        probs = np.full(10, 0.9, dtype=np.float32)

        metrics = compute_metrics_with_ci(labels, predictions, probs)

        assert metrics["_Accuracy_mean"] == 1.0
        assert metrics["_Sensitivity_mean"] == 1.0

    def test_metrics_value_ranges(self, random_predictions):
        """Test that all metrics are in valid ranges."""
        labels, predictions, probs = random_predictions
        metrics = compute_metrics_with_ci(labels, predictions, probs)

        # Check core metrics are in [0, 1]
        core_metrics = [
            "_Accuracy_mean",
            "_AUROC_mean",
            "_Sensitivity_mean",
            "_Specificity_mean",
            "_F1_mean",
        ]
        for metric_name in core_metrics:
            value = metrics.get(metric_name)
            if value is not None and not np.isnan(value):
                assert 0 <= value <= 1, f"{metric_name} out of range: {value}"

    def test_single_sample(self):
        """Test metrics with single sample."""
        labels = np.array([1], dtype=int)
        predictions = np.array([1], dtype=int)
        probs = np.array([0.9], dtype=np.float32)

        # Should handle single sample gracefully (though CI might be empty)
        metrics = compute_metrics_with_ci(labels, predictions, probs)
        assert "Accuracy" in metrics

    def test_imbalanced_labels(self):
        """Test metrics with imbalanced labels."""
        labels = np.array([0] * 90 + [1] * 10, dtype=int)
        predictions = np.array([0] * 85 + [1] * 5 + [0] * 5 + [1] * 5, dtype=int)
        probs = np.random.rand(100).astype(np.float32)

        metrics = compute_metrics_with_ci(labels, predictions, probs)

        # Metrics should be computed even with class imbalance
        assert "F1" in metrics
        assert "AUROC" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
