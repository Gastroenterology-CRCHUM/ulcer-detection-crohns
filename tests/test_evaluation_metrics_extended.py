"""Extended tests for src/evaluation/metrics.py (covers gaps at ~71% coverage)."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    _auroc,
    bootstrap_ci,
    bootstrap_ci_aggregated,
    compute_clip_metrics,
    compute_metrics_with_ci,
    fmt,
)

# ---------------------------------------------------------------------------
# fmt
# ---------------------------------------------------------------------------


class TestFmt:
    def test_nan_lower_returns_ci_unavailable(self):
        result = fmt(0.85, float("nan"), 0.9)
        assert "CI unavailable" in result

    def test_nan_upper_returns_ci_unavailable(self):
        result = fmt(0.85, 0.8, float("nan"))
        assert "CI unavailable" in result

    def test_valid_values_formatted(self):
        result = fmt(0.85, 0.8, 0.9)
        assert "0.85" in result
        assert "0.8" in result
        assert "0.9" in result


# ---------------------------------------------------------------------------
# _auroc — multiclass with missing classes
# ---------------------------------------------------------------------------


class TestAuroc:
    def test_binary_case(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        result = _auroc(labels, None, probs)
        assert 0.0 <= result <= 1.0

    def test_multiclass_all_present(self):
        rng = np.random.default_rng(0)
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = rng.dirichlet(np.ones(3), size=6)
        result = _auroc(labels, None, probs)
        assert 0.0 <= result <= 1.0

    def test_multiclass_missing_class_returns_partial_auroc(self):
        # Only classes 0 and 1 are present from a 3-class model
        labels = np.array([0, 0, 1, 1, 0, 1])
        probs = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.7, 0.2],
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
            ]
        )
        result = _auroc(labels, None, probs)
        assert 0.0 <= result <= 1.0

    def test_multiclass_single_class_returns_nan(self):
        labels = np.array([1, 1, 1, 1])
        probs = np.array([[0.1, 0.8, 0.1]] * 4)
        result = _auroc(labels, None, probs)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# bootstrap_ci — NaN CI path
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_three_values(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        from src.evaluation.metrics import _f1

        mean, lo, hi = bootstrap_ci(labels, preds, probs, _f1, n_bootstrap=100, seed=0)
        assert 0.0 <= mean <= 1.0
        assert lo <= mean <= hi

    def test_single_class_returns_nan_ci(self):
        # All labels the same — bootstrap will always resample one class
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        preds = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        probs = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        from src.evaluation.metrics import _f1

        _, lo, hi = bootstrap_ci(labels, preds, probs, _f1, n_bootstrap=50, seed=0)
        assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# bootstrap_ci_aggregated — NaN path
# ---------------------------------------------------------------------------


class TestBootstrapCIAggregated:
    def test_returns_lower_upper(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = rng.uniform(size=6)
        from sklearn.metrics import roc_auc_score

        lo, hi = bootstrap_ci_aggregated(y_true, y_score, roc_auc_score, n=200, seed=0)
        assert lo <= hi

    def test_single_class_returns_nan(self):
        y_true = np.ones(8, dtype=int)
        y_score = np.ones(8) * 0.9
        from sklearn.metrics import roc_auc_score

        lo, hi = bootstrap_ci_aggregated(y_true, y_score, roc_auc_score, n=50, seed=0)
        assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# compute_metrics_with_ci
# ---------------------------------------------------------------------------


class TestComputeMetricsWithCI:
    def test_returns_all_metric_keys(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        preds = np.array([0, 0, 1, 1, 0, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.7])
        result = compute_metrics_with_ci(labels, preds, probs, n_bootstrap=50)
        for key in ("F1", "Accuracy", "AUROC", "Sensitivity"):
            assert key in result

    def test_raw_mean_keys_present(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        preds = np.array([0, 0, 1, 1, 0, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.7])
        result = compute_metrics_with_ci(labels, preds, probs, n_bootstrap=50)
        assert "_F1_mean" in result
        assert 0.0 <= result["_F1_mean"] <= 1.0


# ---------------------------------------------------------------------------
# compute_clip_metrics
# ---------------------------------------------------------------------------


class TestComputeClipMetrics:
    def test_basic_clip_aggregation(self):
        labels = np.array([0, 0, 1, 1, 1, 0])
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.15])
        video_ids = ["clip_a", "clip_a", "clip_b", "clip_b", "clip_c", "clip_c"]
        result = compute_clip_metrics(labels, probs, video_ids, threshold=0.5)
        assert "F1" in result
        assert "_F1_mean" in result

    def test_custom_threshold(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.3, 0.4, 0.6, 0.7])
        video_ids = ["a", "a", "b", "b"]
        result = compute_clip_metrics(labels, probs, video_ids, threshold=0.45)
        assert "Accuracy" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
