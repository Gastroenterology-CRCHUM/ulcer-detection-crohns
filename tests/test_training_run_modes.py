"""Tests for the pure-logic pieces of src/training/run_modes.py.

The bulk of this module (run_split_mode, run_cv_mode, run_ensemble_inference)
orchestrates real training over real DataLoaders/models/MLflow runs and is
better suited to a lightweight integration test than unit tests with heavy
mocking. This file covers the parts that don't need that: the PipelineDef
dataclass, the pure-numpy branch of _compute_fold_metrics, the
missing-dependency fallback in run_explainability, and the _log_manifest_info
delegation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import src.training.run_modes as run_modes
from src.training.run_modes import PipelineDef, _compute_fold_metrics, _log_manifest_info, run_explainability

# ---------------------------------------------------------------------------
# PipelineDef
# ---------------------------------------------------------------------------


class TestPipelineDef:
    def test_required_fields_are_set(self):
        p = PipelineDef(
            label_col="label",
            num_classes=1,
            models_root=Path("output/ulcer/models"),
            experiment_name="ulcer_detection",
            registry_prefix="ulcer_",
            run_name_infix="",
            aggregate_by_clip=True,
            tune_threshold=True,
            is_multiclass=False,
            pipeline_tag="B_ulcer",
        )
        assert p.label_col == "label"
        assert p.num_classes == 1
        assert p.experiment_name == "ulcer_detection"
        assert p.aggregate_by_clip is True
        assert p.is_multiclass is False

    def test_optional_fields_default(self):
        p = PipelineDef(
            label_col="label",
            num_classes=1,
            models_root=Path("x"),
            experiment_name="e",
            registry_prefix="p_",
            run_name_infix="",
            aggregate_by_clip=False,
            tune_threshold=False,
            is_multiclass=False,
            pipeline_tag="t",
        )
        assert p.tune_clip_threshold is False
        assert p.class_names is None
        assert p.comparison_metrics == ["test__f1_mean", "test__auroc_mean"]
        assert p.comparison_file_suffix == ""
        assert p.extra_tags == {}
        assert p.extra_params == {}

    def test_default_mutable_fields_are_independent_per_instance(self):
        """extra_tags/extra_params/comparison_metrics must not be shared state."""
        p1 = PipelineDef(
            label_col="label", num_classes=1, models_root=Path("x"), experiment_name="e",
            registry_prefix="p_", run_name_infix="", aggregate_by_clip=False,
            tune_threshold=False, is_multiclass=False, pipeline_tag="t1",
        )
        p2 = PipelineDef(
            label_col="label", num_classes=1, models_root=Path("x"), experiment_name="e",
            registry_prefix="p_", run_name_infix="", aggregate_by_clip=False,
            tune_threshold=False, is_multiclass=False, pipeline_tag="t2",
        )
        p1.extra_tags["a"] = "b"
        assert p2.extra_tags == {}


# ---------------------------------------------------------------------------
# _compute_fold_metrics — pure-numpy branch (best_probs/best_labels given)
# ---------------------------------------------------------------------------


def _make_pipeline(is_multiclass: bool) -> PipelineDef:
    return PipelineDef(
        label_col="label",
        num_classes=3 if is_multiclass else 1,
        models_root=Path("x"),
        experiment_name="e",
        registry_prefix="p_",
        run_name_infix="",
        aggregate_by_clip=False,
        tune_threshold=True,
        is_multiclass=is_multiclass,
        pipeline_tag="t",
    )


class TestComputeFoldMetricsBinary:
    def test_slices_2d_probs_to_positive_class_and_thresholds_at_half(self):
        probs_2d = np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3], [0.8, 0.2], [0.4, 0.6]])
        labels = np.array([0, 1, 1, 0, 1])

        result = _compute_fold_metrics(
            model=None,
            pipeline=_make_pipeline(is_multiclass=False),
            best_probs=probs_2d,
            best_labels=labels,
            val_loader=None,
            manifest_path=None,
            device=None,
            fold=2,
        )

        probs_1d = probs_2d[:, 1]
        preds = (probs_1d >= run_modes.CV_THRESHOLD).astype(int)
        exp_p, exp_r, exp_f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        assert result["fold"] == 2
        assert result["val_f1"] == pytest.approx(exp_f1)
        assert result["val_precision"] == pytest.approx(exp_p)
        assert result["val_recall"] == pytest.approx(exp_r)
        assert result["val_auroc"] == pytest.approx(roc_auc_score(labels, probs_1d))
        np.testing.assert_array_equal(result["_probs"], probs_1d)
        np.testing.assert_array_equal(result["_labels"], labels)

    def test_accepts_1d_probs_directly(self):
        probs_1d = np.array([0.9, 0.1, 0.7, 0.8, 0.4])
        labels = np.array([1, 0, 1, 1, 0])

        result = _compute_fold_metrics(
            None, _make_pipeline(is_multiclass=False), probs_1d, labels, None, None, None, fold=0
        )

        preds = (probs_1d >= run_modes.CV_THRESHOLD).astype(int)
        exp_p, exp_r, exp_f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        assert result["val_f1"] == pytest.approx(exp_f1)
        assert result["val_precision"] == pytest.approx(exp_p)
        assert result["val_recall"] == pytest.approx(exp_r)

    def test_nan_auroc_on_single_class_does_not_raise(self):
        probs_1d = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 1, 1])  # single class -> roc_auc_score raises ValueError internally

        result = _compute_fold_metrics(
            None, _make_pipeline(is_multiclass=False), probs_1d, labels, None, None, None, fold=0
        )
        assert np.isnan(result["val_auroc"])


class TestComputeFoldMetricsMulticlass:
    def test_uses_argmax_and_macro_average(self):
        probs = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]
        )
        labels = np.array([0, 1, 2, 0])

        result = _compute_fold_metrics(
            None, _make_pipeline(is_multiclass=True), probs, labels, None, None, None, fold=1
        )

        preds = probs.argmax(axis=1)
        exp_p, exp_r, exp_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        exp_auroc = roc_auc_score(labels, probs, multi_class="ovr")
        assert result["val_f1"] == pytest.approx(exp_f1)
        assert result["val_precision"] == pytest.approx(exp_p)
        assert result["val_recall"] == pytest.approx(exp_r)
        assert result["val_auroc"] == pytest.approx(exp_auroc)


# ---------------------------------------------------------------------------
# run_explainability — missing-dependency fallback
# ---------------------------------------------------------------------------


class TestRunExplainability:
    def test_skips_gracefully_when_visualization_module_missing(self, capsys):
        """src/visualization/ does not exist in this repo — must warn, not raise."""
        result = run_explainability(
            model=None,
            test_loader=None,
            results={},
            device=None,
            results_dir=Path("unused"),
            num_classes=2,
        )
        assert result is None
        assert "Explainability skipped" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _log_manifest_info — delegation to log_dataset_info
# ---------------------------------------------------------------------------


class TestLogManifestInfo:
    def test_delegates_to_log_dataset_info(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(run_modes, "log_dataset_info", lambda p: calls.append(p))

        manifest_path = tmp_path / "manifest.csv"
        _log_manifest_info(manifest_path)

        assert calls == [manifest_path]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
