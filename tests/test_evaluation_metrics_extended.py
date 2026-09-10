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
    patient_clustered_bootstrap,
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


# ---------------------------------------------------------------------------
# patient_clustered_bootstrap
# ---------------------------------------------------------------------------


def _synthetic_clips(rng, n_patients=19, min_clips=2, max_clips=9):
    """Build a (labels, probs_a, probs_b, patient_ids) tuple loosely mirroring
    the real heldout manifest's structure (variable clips/patient, mixed labels)."""
    patient_ids, labels, probs_a, probs_b = [], [], [], []
    for p in range(n_patients):
        n_clips = rng.integers(min_clips, max_clips + 1)
        pos_frac = rng.uniform(0.2, 0.8)
        for _ in range(n_clips):
            lbl = int(rng.uniform() < pos_frac)
            patient_ids.append(f"patient_{p}")
            labels.append(lbl)
            # Deliberately noisy/overlapping (AUROC well below 1.0) so the
            # bootstrap distribution has genuine spread to test against.
            probs_a.append(np.clip(rng.normal(0.5 + 0.15 * (2 * lbl - 1), 0.28), 0.01, 0.99))
            probs_b.append(np.clip(rng.normal(0.5 + 0.15 * (2 * lbl - 1), 0.28), 0.01, 0.99))
    return (
        np.array(labels),
        np.array(probs_a),
        np.array(probs_b),
        np.array(patient_ids),
    )


class TestPatientClusteredBootstrap:
    def test_identical_models_zero_point_delta(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(0)
        labels, probs_a, _, patient_ids = _synthetic_clips(rng)

        def statistic_fn(w):
            auc = roc_auc_score(labels, probs_a, sample_weight=w)
            return auc - auc  # same model twice -> exactly 0 every time

        result = patient_clustered_bootstrap(
            labels, patient_ids, statistic_fn, n_bootstrap=500, seed=1
        )
        assert result["point"] == 0.0
        lo, hi = result["ci_percentile"]
        assert lo <= 0.0 <= hi
        # Symmetric-around-zero, not just containing zero
        assert abs((lo + hi) / 2) < 0.02

    def test_point_estimate_is_plugin_not_bootstrap_mean(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(2)
        labels, probs_a, _, patient_ids = _synthetic_clips(rng)

        def statistic_fn(w):
            return roc_auc_score(labels, probs_a, sample_weight=w)

        plugin = roc_auc_score(labels, probs_a)
        result = patient_clustered_bootstrap(
            labels, patient_ids, statistic_fn, n_bootstrap=500, seed=3
        )
        # The reported point estimate must be the plug-in value computed once
        # on the full data — not derived from the replicate distribution at all.
        assert result["point"] == pytest.approx(plugin)

    def test_degenerate_resamples_are_dropped_and_counted(self):
        # 3 patients: two are label-pure, only draws mixing them both avoid
        # a single-class resample -> plenty of degenerate draws expected.
        labels = np.array([1, 1, 0, 1])
        patient_ids = np.array(["A", "A", "B", "C"])  # A: pure positive, B: pure negative, C: positive
        probs = np.array([0.9, 0.8, 0.2, 0.7])

        def statistic_fn(w):
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(labels, probs, sample_weight=w)

        result = patient_clustered_bootstrap(
            labels, patient_ids, statistic_fn, n_bootstrap=300, seed=4
        )
        assert result["n_dropped"] > 0
        assert result["n_dropped"] + len(result["replicates"]) == 300

    def test_paired_delta_is_narrower_than_independent_differencing(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(5)
        labels, probs_a, probs_b, patient_ids = _synthetic_clips(rng)
        # Make B a noisy copy of A so the two are correlated, as in the
        # real use case (two model configs scored on the same clips).
        probs_b = np.clip(probs_a + rng.normal(0, 0.03, size=len(probs_a)), 0.01, 0.99)

        def delta_fn(w):
            return roc_auc_score(labels, probs_a, sample_weight=w) - roc_auc_score(
                labels, probs_b, sample_weight=w
            )

        paired = patient_clustered_bootstrap(
            labels, patient_ids, delta_fn, n_bootstrap=2000, seed=6
        )

        def auc_a_fn(w):
            return roc_auc_score(labels, probs_a, sample_weight=w)

        def auc_b_fn(w):
            return roc_auc_score(labels, probs_b, sample_weight=w)

        indep_a = patient_clustered_bootstrap(
            labels, patient_ids, auc_a_fn, n_bootstrap=2000, seed=7
        )
        indep_b = patient_clustered_bootstrap(
            labels, patient_ids, auc_b_fn, n_bootstrap=2000, seed=8
        )
        paired_sd = float(np.std(paired["replicates"]))
        indep_sd = float(
            np.sqrt(np.var(indep_a["replicates"]) + np.var(indep_b["replicates"]))
        )
        assert paired_sd < indep_sd

    def test_bca_diverges_from_percentile_under_skew(self):
        # Few, unevenly-sized clusters with a near-boundary (but not
        # perfectly separable — overlapping score ranges) statistic tend
        # to produce a skewed bootstrap distribution.
        rng = np.random.default_rng(9)
        n_patients = 6
        patient_ids, labels, probs = [], [], []
        for p in range(n_patients):
            n_clips = 1 if p < 4 else 6
            for _ in range(n_clips):
                lbl = int(p >= 4 or rng.uniform() < 0.15)
                patient_ids.append(f"p{p}")
                labels.append(lbl)
                probs.append(
                    np.clip(rng.uniform(0.55, 0.95) if lbl else rng.uniform(0.05, 0.6), 0, 1)
                )
        labels, probs, patient_ids = np.array(labels), np.array(probs), np.array(patient_ids)

        def statistic_fn(w):
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(labels, probs, sample_weight=w)

        result = patient_clustered_bootstrap(
            labels, patient_ids, statistic_fn, n_bootstrap=3000, seed=10
        )
        assert result["ci_bca"] != result["ci_percentile"]

    def test_empty_replicates_returns_nan_ci(self):
        # Every patient is entirely one label -> every resample is degenerate.
        labels = np.array([1, 1, 1, 1])
        patient_ids = np.array(["A", "A", "B", "B"])
        probs = np.array([0.9, 0.8, 0.7, 0.6])

        def statistic_fn(w):
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(labels, probs, sample_weight=w)

        result = patient_clustered_bootstrap(
            labels, patient_ids, statistic_fn, n_bootstrap=50, seed=11
        )
        assert result["n_dropped"] == 50
        assert len(result["replicates"]) == 0
        assert np.isnan(result["ci_percentile"][0])
        assert np.isnan(result["ci_bca"][0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
