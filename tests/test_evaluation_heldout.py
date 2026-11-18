"""Tests for the held-out analysis modules: heldout, bootstrap (statistics),
mlflow_io (run selection), heldout_tables (manuscript tables), reporting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.bootstrap import (
    delta_auroc_statistic,
    flip_delta,
    metric_statistic,
    patient_clustered_bootstrap,
)
from src.evaluation.heldout import (
    aggregate_to_clip,
    load_cohort,
    load_informative_mask,
    point_metrics,
)
from src.evaluation.heldout_tables import (
    PAPER_COMPARISONS,
    _paper_row_header,
    paper_comparisons,
)
from src.evaluation.mlflow_io import load_cv_runs
from src.evaluation.reporting import write_table

# ---------------------------------------------------------------------------
# Cohort, aggregation, point metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest(tmp_path):
    """3 clips over 2 patients; clip c2 has a 2-vs-1 label vote."""
    df = pd.DataFrame(
        {
            "relative_path": [f"f{i}.png" for i in range(7)],
            "clip_key": ["c2", "c2", "c2", "c1", "c1", "c3", "c3"],
            "patient_id": ["p1", "p1", "p1", "p1", "p1", "p2", "p2"],
            "label": [1, 1, 0, 0, 0, 1, 1],
        }
    )
    path = tmp_path / "manifest.csv"
    df.to_csv(path, index=False)
    return path


class TestCohort:
    def test_clips_sorted_with_majority_label(self, manifest):
        cohort = load_cohort(manifest)
        assert cohort.clip_order == ["c1", "c2", "c3"]
        assert cohort.clip_labels.tolist() == [0, 1, 1]
        assert cohort.clip_patient_ids.tolist() == ["p1", "p1", "p2"]
        assert len(cohort.frame_clip_keys) == 7

    def test_mask_drops_frames_then_revotes_and_drops_empty_clips(self, manifest):
        mask = np.array([False, False, True, True, True, True, True])  # c2 keeps its label-0 frame
        cohort = load_cohort(manifest, mask)
        assert cohort.clip_labels_in(["c2"]).tolist() == [0]
        mask = np.array([True, True, True, False, False, True, True])  # c1 loses every frame
        assert load_cohort(manifest, mask).clip_order == ["c2", "c3"]

    def test_informative_mask_keeps_unknown_frames(self, manifest, tmp_path):
        mask_path = tmp_path / "mask.csv"
        pd.DataFrame({"relative_path": ["f0.png", "f1.png"], "informative": [0, 1]}).to_csv(
            mask_path, index=False
        )
        keep = load_informative_mask(manifest, mask_path)
        assert keep.tolist() == [False, True, True, True, True, True, True]


def test_aggregate_to_clip_mean_in_requested_order():
    keys = np.array(["b", "a", "b", "a", "c"])
    probs = np.array([0.2, 0.9, 0.4, 0.7, 0.5])
    np.testing.assert_allclose(aggregate_to_clip(probs, keys, ["c", "a", "b"]), [0.5, 0.8, 0.3])


def test_point_metrics_known_values():
    labels = np.array([1, 1, 1, 0, 0])
    probs = np.array([0.9, 0.6, 0.2, 0.7, 0.1])
    m = point_metrics(labels, probs, threshold=0.5)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["auroc"] == pytest.approx(roc_auc_score(labels, probs))


# ---------------------------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------------------------


def _clustered(seed=0, n_patients=15):
    rng = np.random.default_rng(seed)
    labels, a, b, pids = [], [], [], []
    for p in range(n_patients):
        for _ in range(rng.integers(2, 6)):
            y = int(rng.uniform() < 0.5)
            labels.append(y)
            a.append(np.clip(rng.normal(0.5 + 0.2 * (2 * y - 1), 0.25), 0, 1))
            b.append(np.clip(rng.normal(0.5 + 0.1 * (2 * y - 1), 0.25), 0, 1))
            pids.append(f"p{p}")
    return np.array(labels), np.array(a), np.array(b), np.array(pids)


def test_flip_delta_matches_bootstrapping_the_reversed_difference():
    labels, a, b, pids = _clustered()
    kw = {"n_bootstrap": 1000, "seed": 3}
    ab = patient_clustered_bootstrap(labels, pids, delta_auroc_statistic(labels, a, b), **kw)
    ba = patient_clustered_bootstrap(labels, pids, delta_auroc_statistic(labels, b, a), **kw)
    point, ci = flip_delta(ab["point"], ab["ci_bca"])
    assert point == pytest.approx(ba["point"], abs=1e-12)
    np.testing.assert_allclose(ci, ba["ci_bca"], atol=1e-9)


def test_metric_statistic_full_weights_equals_point_metric():
    labels, a, _, _ = _clustered(seed=1)
    ones = np.ones(len(labels))
    expected = point_metrics(labels, a, 0.5)
    for metric in ("auroc", "f1", "precision", "sensitivity", "specificity", "accuracy"):
        assert metric_statistic(metric, labels, a, 0.5)(ones) == pytest.approx(expected[metric])
    with pytest.raises(ValueError):
        metric_statistic("npv", labels, a, 0.5)


# ---------------------------------------------------------------------------
# MLflow run selection
# ---------------------------------------------------------------------------


def _run(run_id, start, tags, params=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start),
        data=SimpleNamespace(tags=tags, params=params or {}, metrics={}),
    )


class _FakeClient:
    def __init__(self, runs):
        self._runs = runs

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1") if name == "exp" else None

    def search_runs(self, experiment_ids, max_results):
        return self._runs


def test_load_cv_runs_latest_cv_parent_and_sorted_folds(capsys):
    cv = {"training_mode": "cv_5fold", "model": "efficientnetb0"}
    runs = [
        _run("old", 1, cv),
        _run("new", 2, cv),
        _run("split", 3, {"training_mode": "split", "model": "efficientnetb0"}),
        _run("k1", 4, {"mlflow.parentRunId": "new"}, {"fold": "1"}),
        _run("k0", 5, {"mlflow.parentRunId": "new"}, {"fold": "0"}),
        _run("x0", 6, {"mlflow.parentRunId": "old"}, {"fold": "0"}),
    ]
    out = load_cv_runs(_FakeClient(runs), "exp")
    assert list(out) == ["efficientnetb0"]
    assert out["efficientnetb0"].parent.info.run_id == "new"
    assert [r.info.run_id for r in out["efficientnetb0"].children] == ["k0", "k1"]
    assert "2 CV parent runs" in capsys.readouterr().out
    with pytest.raises(ValueError):
        load_cv_runs(_FakeClient(runs), "missing")


# ---------------------------------------------------------------------------
# Manuscript tables and writers
# ---------------------------------------------------------------------------


def test_paper_row_header_marks_self_supervised():
    assert _paper_row_header("efficientnetb0") == {
        "Backbone": "EfficientNet-B0",
        "Pretraining": "ImageNet-1K",
    }
    assert _paper_row_header("resnet50_gastronet")["Pretraining"] == "GastroNet-5M (DINOv1) †"


def test_paper_comparisons_are_oriented_as_written():
    keys = sorted({k for c in PAPER_COMPARISONS for k in (c.a, c.b)})
    labels = np.array([0, 1, 0, 1])
    configs = {k: SimpleNamespace(ensemble_clip_probs=np.array([0.1, 0.9, 0.2, 0.8])) for k in keys}
    cohort = SimpleNamespace(clip_labels=labels)
    # Store every pair in alphabetical order with delta = +0.05 for (first, second).
    boots = {(a, b): {"point": 0.05, "ci_bca": (-0.01, 0.12)} for a in keys for b in keys if a < b}
    df = paper_comparisons(boots, configs, cohort).set_index("comparison")
    for c in PAPER_COMPARISONS:
        stored_as_written = c.a < c.b
        expected = "+0.0500" if stored_as_written else "-0.0500"
        expected_ci = "-0.0100 to +0.1200" if stored_as_written else "-0.1200 to +0.0100"
        assert df.loc[c.label, "delta_auroc"] == expected
        assert df.loc[c.label, "ci_95_bca"] == expected_ci


def test_write_table_writes_csv_and_markdown(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x ±y", "z"]})
    write_table(df, tmp_path / "t")
    assert pd.read_csv(tmp_path / "t.csv").equals(df)
    md = (tmp_path / "t.md").read_text(encoding="utf-8")
    assert md.splitlines()[0] == "| a | b |"
    assert "x ±y" in md
