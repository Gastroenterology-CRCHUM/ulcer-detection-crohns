"""Tests for src/noninformative/model.py."""

import numpy as np
import pytest

from src.noninformative.model import DEFAULT_RF_PARAMS, NonInformativeClassifier

# ---------------------------------------------------------------------------
# Fixture: tiny synthetic dataset (fast RF training)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_dataset():
    """20 samples, 10 features, balanced binary labels."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 10)).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10)
    feat_names = [f"feat_{i}" for i in range(10)]
    return X, y, feat_names


@pytest.fixture
def fitted_clf(tiny_dataset):
    X, y, names = tiny_dataset
    clf = NonInformativeClassifier(rf_params={"n_estimators": 5, "random_state": 0, "n_jobs": 1})
    clf.fit(X, y, feature_names=names, verbose=False)
    return clf, X, y


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults_applied(self):
        clf = NonInformativeClassifier()
        for k, v in DEFAULT_RF_PARAMS.items():
            assert clf.rf_params[k] == v

    def test_custom_params_merged(self):
        clf = NonInformativeClassifier(rf_params={"n_estimators": 10})
        assert clf.rf_params["n_estimators"] == 10

    def test_not_fitted_initially(self):
        clf = NonInformativeClassifier()
        assert clf.rf is None
        assert clf.scaler is None

    def test_default_threshold(self):
        assert NonInformativeClassifier().threshold == 0.5


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_returns_self(self, tiny_dataset):
        X, y, _ = tiny_dataset
        clf = NonInformativeClassifier(rf_params={"n_estimators": 3, "n_jobs": 1})
        result = clf.fit(X, y, verbose=False)
        assert result is clf

    def test_rf_and_scaler_set(self, fitted_clf):
        clf, _, _ = fitted_clf
        assert clf.rf is not None
        assert clf.scaler is not None

    def test_feature_importances_set(self, fitted_clf):
        clf, _, _ = fitted_clf
        assert clf.feature_importances is not None
        assert len(clf.feature_importances) == 10

    def test_feature_importances_not_set_without_names(self, tiny_dataset):
        X, y, _ = tiny_dataset
        clf = NonInformativeClassifier(rf_params={"n_estimators": 3, "n_jobs": 1})
        clf.fit(X, y, feature_names=None, verbose=False)
        assert clf.feature_importances is None

    def test_check_fitted_raises_before_fit(self):
        clf = NonInformativeClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(np.zeros((1, 5)))


# ---------------------------------------------------------------------------
# predict_proba / predict / predict_single
# ---------------------------------------------------------------------------


class TestInference:
    def test_predict_proba_shape(self, fitted_clf):
        clf, X, _ = fitted_clf
        probs = clf.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, fitted_clf):
        clf, X, _ = fitted_clf
        probs = clf.predict_proba(X)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_returns_binary(self, fitted_clf):
        clf, X, _ = fitted_clf
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_single_returns_tuple(self, fitted_clf):
        clf, X, _ = fitted_clf
        label, prob = clf.predict_single(X[0])
        assert label in (0, 1)
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# tune_threshold
# ---------------------------------------------------------------------------


class TestTuneThreshold:
    def test_returns_float_in_range(self, fitted_clf):
        clf, X, y = fitted_clf
        t = clf.tune_threshold(X, y, metric="f1", n_steps=10, verbose=False)
        assert 0.0 < t < 1.0

    def test_stores_threshold(self, fitted_clf):
        clf, X, y = fitted_clf
        t = clf.tune_threshold(X, y, metric="f1", n_steps=10, verbose=False)
        assert clf.threshold == t

    def test_balanced_accuracy_metric(self, fitted_clf):
        clf, X, y = fitted_clf
        t = clf.tune_threshold(X, y, metric="balanced_accuracy", n_steps=10, verbose=False)
        assert 0.0 < t < 1.0

    def test_accuracy_metric_fallback(self, fitted_clf):
        clf, X, y = fitted_clf
        t = clf.tune_threshold(X, y, metric="accuracy", n_steps=10, verbose=False)
        assert 0.0 < t < 1.0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_expected_keys(self, fitted_clf):
        clf, X, y = fitted_clf
        results = clf.evaluate(X, y, n_bootstrap=50)
        for key in ("f1", "roc_auc", "accuracy", "sensitivity", "specificity", "confusion_matrix"):
            assert key in results

    def test_threshold_in_results(self, fitted_clf):
        clf, X, y = fitted_clf
        results = clf.evaluate(X, y, n_bootstrap=50)
        assert results["threshold"] == clf.threshold

    def test_predictions_shape(self, fitted_clf):
        clf, X, y = fitted_clf
        results = clf.evaluate(X, y, n_bootstrap=50)
        assert len(results["predictions"]) == len(y)
        assert len(results["probabilities"]) == len(y)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_file(self, fitted_clf, tmp_path):
        clf, _, _ = fitted_clf
        path = tmp_path / "model.pkl"
        clf.save(path)
        assert path.exists()

    def test_load_restores_threshold(self, fitted_clf, tmp_path):
        clf, X, y = fitted_clf
        clf.tune_threshold(X, y, metric="f1", n_steps=10, verbose=False)
        path = tmp_path / "model.pkl"
        clf.save(path)
        loaded = NonInformativeClassifier.load(path)
        assert loaded.threshold == pytest.approx(clf.threshold)

    def test_load_restores_feature_importances(self, fitted_clf, tmp_path):
        clf, _, _ = fitted_clf
        path = tmp_path / "model.pkl"
        clf.save(path)
        loaded = NonInformativeClassifier.load(path)
        assert loaded.feature_importances is not None
        assert len(loaded.feature_importances) == 10

    def test_loaded_model_predicts(self, fitted_clf, tmp_path):
        clf, X, _ = fitted_clf
        path = tmp_path / "model.pkl"
        clf.save(path)
        loaded = NonInformativeClassifier.load(path)
        preds = loaded.predict(X)
        assert len(preds) == len(X)

    def test_save_creates_parent_dirs(self, fitted_clf, tmp_path):
        clf, _, _ = fitted_clf
        path = tmp_path / "deep" / "nested" / "model.pkl"
        clf.save(path)
        assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
