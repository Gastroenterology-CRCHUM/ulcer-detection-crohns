"""Tests for src/evaluation/delong.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.delong import _placement_values, delong_matrix, delong_test


class TestPlacementValues:
    def test_perfect_classifier(self):
        """All positives > all negatives → AUC = 1."""
        labels = np.array([1, 1, 0, 0])
        probs = np.array([0.9, 0.8, 0.2, 0.1])
        V10, V01 = _placement_values(labels, probs)
        assert np.allclose(V10.mean(), 1.0)
        assert np.allclose(V01.mean(), 1.0)

    def test_worst_classifier(self):
        """All positives < all negatives → AUC = 0."""
        labels = np.array([1, 1, 0, 0])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        V10, V01 = _placement_values(labels, probs)
        assert np.allclose(V10.mean(), 0.0)
        assert np.allclose(V01.mean(), 0.0)

    def test_tie_breaking(self):
        """Tied pos==neg contributes 0.5 × 1/n0."""
        labels = np.array([1, 0])
        probs = np.array([0.5, 0.5])
        V10, V01 = _placement_values(labels, probs)
        assert V10[0] == pytest.approx(0.5)
        assert V01[0] == pytest.approx(0.5)

    def test_shapes(self):
        """V10 shape = n_positives, V01 shape = n_negatives."""
        labels = np.array([1, 1, 1, 0, 0])
        probs = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        V10, V01 = _placement_values(labels, probs)
        assert V10.shape == (3,)
        assert V01.shape == (2,)

    def test_auc_consistency(self):
        """mean(V10) == mean(V01) (both equal the AUC)."""
        np.random.seed(5)
        labels = np.random.randint(0, 2, 40)
        probs = np.random.rand(40)
        V10, V01 = _placement_values(labels, probs)
        assert V10.mean() == pytest.approx(V01.mean(), abs=1e-10)


class TestDelongTest:
    def test_identical_models_returns_zero(self):
        """Identical models → z=0, p=1."""
        labels = np.array([1, 1, 0, 0])
        probs = np.array([0.9, 0.8, 0.2, 0.1])
        auc_a, auc_b, z, p = delong_test(labels, probs, probs)
        assert z == pytest.approx(0.0)
        assert p == pytest.approx(1.0)
        assert auc_a == pytest.approx(auc_b)

    def test_significant_difference(self):
        """Clearly different models → p < 0.05."""
        np.random.seed(0)
        labels = np.array([1] * 100 + [0] * 100)
        probs_a = np.concatenate(
            [
                np.random.uniform(0.7, 1.0, 100),
                np.random.uniform(0.0, 0.3, 100),
            ]
        )
        probs_b = np.concatenate(
            [
                np.random.uniform(0.3, 0.7, 100),
                np.random.uniform(0.3, 0.7, 100),
            ]
        )
        auc_a, auc_b, z, p = delong_test(labels, probs_a, probs_b)
        assert auc_a > auc_b
        assert p < 0.05

    def test_return_types_are_float(self):
        """All four return values are Python floats."""
        labels = np.array([1, 0, 1, 0])
        probs_a = np.array([0.9, 0.1, 0.8, 0.2])
        probs_b = np.array([0.7, 0.3, 0.6, 0.4])
        result = delong_test(labels, probs_a, probs_b)
        assert all(isinstance(v, float) for v in result)

    def test_p_value_in_range(self):
        """p-value always in [0, 1]."""
        np.random.seed(99)
        labels = np.random.randint(0, 2, 50)
        probs_a = np.random.rand(50)
        probs_b = np.random.rand(50)
        _, _, _, p = delong_test(labels, probs_a, probs_b)
        assert 0.0 <= p <= 1.0

    def test_accepts_list_inputs(self):
        """Lists are coerced to ndarray without error."""
        auc_a, auc_b, z, p = delong_test([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], [0.8, 0.7, 0.3, 0.2])
        assert 0.0 <= p <= 1.0

    def test_z_stat_sign(self):
        """z > 0 when AUC_A > AUC_B."""
        labels = np.array([1, 1, 0, 0, 1, 0] * 10)
        probs_a = np.array([0.9, 0.85, 0.15, 0.1, 0.8, 0.2] * 10)
        probs_b = np.array([0.6, 0.55, 0.45, 0.4, 0.5, 0.5] * 10)
        auc_a, auc_b, z, _ = delong_test(labels, probs_a, probs_b)
        assert auc_a > auc_b
        assert z > 0


class TestDelongMatrix:
    @pytest.fixture
    def two_models(self):
        labels = np.array([1, 1, 0, 0, 1, 0] * 5)
        return {
            "labels": labels,
            "probs": {
                "A": np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.15] * 5),
                "B": np.array([0.6, 0.7, 0.4, 0.3, 0.65, 0.35] * 5),
            },
        }

    def test_two_models_structure(self, two_models):
        p_matrix, df = delong_matrix(two_models["labels"], two_models["probs"])
        assert p_matrix.shape == (2, 2)
        assert set(p_matrix.index) == {"A", "B"}
        required_cols = {
            "Model A",
            "Model B",
            "AUC A",
            "AUC B",
            "ΔAUC",
            "z",
            "p-value",
            "significant",
        }
        assert required_cols.issubset(df.columns)

    def test_two_models_one_row(self, two_models):
        _, df = delong_matrix(two_models["labels"], two_models["probs"])
        assert len(df) == 1

    def test_three_models_three_rows(self):
        labels = np.array([1, 1, 0, 0, 1, 0] * 5)
        model_probs = {
            "A": np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.15] * 5),
            "B": np.array([0.6, 0.7, 0.4, 0.3, 0.65, 0.35] * 5),
            "C": np.array([0.55, 0.6, 0.45, 0.5, 0.58, 0.42] * 5),
        }
        _, df = delong_matrix(labels, model_probs)
        assert len(df) == 3

    def test_significance_flag_is_bool(self, two_models):
        _, df = delong_matrix(two_models["labels"], two_models["probs"])
        assert df["significant"].dtype == bool

    def test_sorted_by_p_value(self):
        np.random.seed(7)
        labels = np.array([1] * 50 + [0] * 50)
        model_probs = {
            "A": np.concatenate([np.random.uniform(0.7, 1.0, 50), np.random.uniform(0.0, 0.3, 50)]),
            "B": np.concatenate([np.random.uniform(0.6, 0.9, 50), np.random.uniform(0.1, 0.4, 50)]),
            "C": np.concatenate([np.random.uniform(0.5, 0.8, 50), np.random.uniform(0.2, 0.5, 50)]),
        }
        _, df = delong_matrix(labels, model_probs)
        assert list(df["p-value"]) == sorted(df["p-value"])

    def test_upper_triangle_only(self, two_models):
        """Diagonal and lower triangle are NaN; upper triangle is filled."""
        p_matrix, _ = delong_matrix(two_models["labels"], two_models["probs"])
        assert pd.isna(p_matrix.loc["A", "A"])
        assert pd.isna(p_matrix.loc["B", "A"])
        assert not pd.isna(p_matrix.loc["A", "B"])

    def test_custom_alpha(self, two_models):
        """Looser alpha classifies more pairs as significant."""
        _, df_strict = delong_matrix(two_models["labels"], two_models["probs"], alpha=0.001)
        _, df_loose = delong_matrix(two_models["labels"], two_models["probs"], alpha=0.99)
        assert df_loose["significant"].sum() >= df_strict["significant"].sum()

    def test_delta_auc_sign(self, two_models):
        """ΔAUC column equals AUC A - AUC B."""
        _, df = delong_matrix(two_models["labels"], two_models["probs"])
        row = df.iloc[0]
        assert row["ΔAUC"] == pytest.approx(row["AUC A"] - row["AUC B"], abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
