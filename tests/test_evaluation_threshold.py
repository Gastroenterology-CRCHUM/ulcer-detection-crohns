"""
tests/test_evaluation_threshold.py
===================================
Unit tests for threshold optimization.
"""

import numpy as np
import pytest
import torch

from src.evaluation.threshold import collect_probabilities, find_best_threshold, sweep_thresholds


class TestSweepThresholds:
    """Test sweep_thresholds function."""

    def test_sweep_basic(self, sample_binary_arrays):
        """Test basic threshold sweep."""
        probs, labels = sample_binary_arrays
        results = sweep_thresholds(probs, labels, n_thresholds=10)

        assert isinstance(results, list)
        assert len(results) == 11  # n_thresholds + 1
        # Each result should be a dict
        assert all(isinstance(r, dict) for r in results)

    def test_sweep_result_structure(self, sample_binary_arrays):
        """Test that sweep results have required fields."""
        probs, labels = sample_binary_arrays
        results = sweep_thresholds(probs, labels, n_thresholds=5)

        required_keys = {
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "confusion_matrix",
        }
        for result in results:
            assert required_keys.issubset(result.keys())

    def test_sweep_threshold_range(self, sample_binary_arrays):
        """Test that thresholds are in expected range."""
        probs, labels = sample_binary_arrays
        results = sweep_thresholds(probs, labels, n_thresholds=99)

        thresholds = [r["threshold"] for r in results]
        # Thresholds are evenly spaced in [0.1, 0.9]
        assert min(thresholds) >= 0.1
        assert max(thresholds) <= 0.9

    def test_sweep_metrics_valid_range(self, sample_binary_arrays):
        """Test that metrics are in valid ranges."""
        probs, labels = sample_binary_arrays
        results = sweep_thresholds(probs, labels, n_thresholds=10)

        for result in results:
            assert 0 <= result["accuracy"] <= 1
            assert 0 <= result["precision"] <= 1
            assert 0 <= result["recall"] <= 1
            assert 0 <= result["f1"] <= 1
            assert 0 <= result["roc_auc"] <= 1

    def test_sweep_different_thresholds_count(self, sample_binary_arrays):
        """Test sweep with different numbers of thresholds."""
        probs, labels = sample_binary_arrays

        for n in [5, 50, 100]:
            results = sweep_thresholds(probs, labels, n_thresholds=n)
            assert len(results) == n + 1

    def test_sweep_perfect_predictions(self):
        """Test sweep with perfect predictions."""
        probs = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=int)

        results = sweep_thresholds(probs, labels, n_thresholds=10)
        # At some threshold, should get perfect accuracy
        assert any(r["accuracy"] == 1.0 for r in results)

    def test_sweep_random_predictions(self):
        """Test sweep with mostly random predictions."""
        np.random.seed(42)
        probs = np.random.rand(100).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(int)

        results = sweep_thresholds(probs, labels, n_thresholds=20)
        # Even random predictions should give consistent metrics
        assert all(0 <= r["f1"] <= 1 for r in results)

    def test_sweep_imbalanced_labels(self):
        """Test sweep with imbalanced labels."""
        probs = np.concatenate(
            [
                np.random.rand(80),  # Majority class
                np.random.rand(20) + 0.5,  # Minority class offset
            ]
        ).astype(np.float32)
        labels = np.concatenate([np.zeros(80, dtype=int), np.ones(20, dtype=int)])

        results = sweep_thresholds(probs, labels, n_thresholds=10)
        assert len(results) > 0


class TestFindBestThreshold:
    """Test find_best_threshold function."""

    def test_find_best_f1(self, sample_threshold_results):
        """Test finding best threshold by F1 score."""
        best = find_best_threshold(sample_threshold_results, metric="f1")
        assert best["threshold"] == 0.4
        assert best["f1"] == 0.75

    def test_find_best_accuracy(self, sample_threshold_results):
        """Test finding best threshold by accuracy."""
        best = find_best_threshold(sample_threshold_results, metric="accuracy")
        assert best["threshold"] == 0.4
        assert best["accuracy"] == 0.80

    def test_find_best_precision(self, sample_threshold_results):
        """Test finding best threshold by precision."""
        best = find_best_threshold(sample_threshold_results, metric="precision")
        assert best["threshold"] == 0.4
        assert best["precision"] == 0.82

    def test_find_best_invalid_metric(self, sample_threshold_results):
        """Test error on invalid metric."""
        with pytest.raises(KeyError):
            find_best_threshold(sample_threshold_results, metric="invalid_metric")

    def test_find_best_empty_results(self):
        """Test error on empty results."""
        with pytest.raises(ValueError):
            find_best_threshold([], metric="f1")

    def test_find_best_single_result(self):
        """Test finding best from single result."""
        results = [{"threshold": 0.5, "f1": 0.8}]
        best = find_best_threshold(results, metric="f1")
        assert best["threshold"] == 0.5


class TestCollectProbabilities:
    """Test collect_probabilities function."""

    def test_sigmoid_binary_model(self):
        """collect_probabilities returns (N,) arrays for a single-output (sigmoid) model."""

        class FixedLogitModel(torch.nn.Module):
            """Returns a constant logit of 2.0 for every sample."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.full((x.shape[0], 1), 2.0)

        images = torch.zeros(6, 3, 4, 4)
        labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images, labels), batch_size=3
        )

        probs, out_labels = collect_probabilities(
            FixedLogitModel(), loader, device=torch.device("cpu"), num_classes=1
        )

        expected_prob = float(torch.sigmoid(torch.tensor(2.0)))
        assert probs.shape == (6,)
        assert out_labels.shape == (6,)
        assert np.allclose(probs, expected_prob, atol=1e-5)
        assert list(out_labels) == [0, 1, 0, 1, 0, 1]

    def test_softmax_two_class_model(self):
        """collect_probabilities returns class-1 probs for a two-output (softmax) model."""

        class FixedTwoClassModel(torch.nn.Module):
            """Returns fixed logits [0.0, 1.0] for every sample."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 1.0]]).expand(x.shape[0], -1)

        images = torch.zeros(4, 3, 4, 4)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images, labels), batch_size=4
        )

        probs, out_labels = collect_probabilities(
            FixedTwoClassModel(), loader, device=torch.device("cpu"), num_classes=2
        )

        expected_prob = float(torch.softmax(torch.tensor([0.0, 1.0]), dim=0)[1])
        assert probs.shape == (4,)
        assert np.allclose(probs, expected_prob, atol=1e-5)
        assert list(out_labels) == [0, 1, 0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
