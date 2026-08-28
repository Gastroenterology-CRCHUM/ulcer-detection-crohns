"""Tests for src/evaluation/mlflow_utils.py.

Runs against a real MLflow tracking store (per-test sqlite DB via the
mlflow_tmp_tracking fixture) rather than mocks — MLflow's local-file testing
story is solid, and exercising the real client catches API-shape drift that
a mock would hide.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from mlflow import MlflowClient

from src.evaluation.mlflow_utils import (
    compare_runs_to_markdown,
    get_best_run,
    get_champion,
    log_ci_artifact,
    log_confusion_matrix,
    log_dataset_info,
    log_figures,
    log_figures_from_dir,
    log_split_metrics,
    promote_model,
    register_best_model,
    set_run_tags,
)

# ---------------------------------------------------------------------------
# get_champion — pure string formatting, no MLflow calls
# ---------------------------------------------------------------------------


class TestGetChampion:
    def test_formats_model_uri(self):
        assert get_champion("ulcer_vits16_gastronet") == "models:/ulcer_vits16_gastronet@champion"


# ---------------------------------------------------------------------------
# set_run_tags
# ---------------------------------------------------------------------------


class TestSetRunTags:
    def test_sets_standard_and_extra_tags(self, mlflow_tmp_tracking):
        with mlflow.start_run() as run:
            set_run_tags("vits16_gastronet", "cv_5fold", {"pipeline": "B_ulcer"})
            run_id = run.info.run_id

        tags = MlflowClient().get_run(run_id).data.tags
        assert tags["model"] == "vits16_gastronet"
        assert tags["training_mode"] == "cv_5fold"
        assert tags["pipeline"] == "B_ulcer"
        expected_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        assert tags["gpu"] == expected_gpu

    def test_extra_tags_are_stringified(self, mlflow_tmp_tracking):
        with mlflow.start_run() as run:
            set_run_tags("m", "split", {"freeze_layers": 0})
            run_id = run.info.run_id

        assert MlflowClient().get_run(run_id).data.tags["freeze_layers"] == "0"


# ---------------------------------------------------------------------------
# log_split_metrics
# ---------------------------------------------------------------------------


class TestLogSplitMetrics:
    def test_only_mean_keys_are_logged_with_split_prefix(self, mlflow_tmp_tracking):
        metrics = {
            "F1": "0.876 (0.851-0.901)",
            "_F1_mean": 0.876,
            "_F1_lower": 0.851,
            "_F1_upper": 0.901,
            "_AUROC_mean": 0.912,
        }
        with mlflow.start_run() as run:
            log_split_metrics(metrics, split="test")
            run_id = run.info.run_id

        logged = MlflowClient().get_run(run_id).data.metrics
        assert logged == {"test__f1_mean": 0.876, "test__auroc_mean": 0.912}

    def test_empty_means_logs_nothing(self, mlflow_tmp_tracking):
        with mlflow.start_run() as run:
            log_split_metrics({"F1": "n/a"}, split="test")
            run_id = run.info.run_id

        assert MlflowClient().get_run(run_id).data.metrics == {}


# ---------------------------------------------------------------------------
# log_ci_artifact
# ---------------------------------------------------------------------------


class TestLogCiArtifact:
    def test_writes_ci_bounds_as_json_artifact(self, mlflow_tmp_tracking, tmp_path):
        metrics = {
            "_F1_mean": 0.876,
            "_F1_lower": 0.851,
            "_F1_upper": 0.901,
            "_AUROC_mean": float("nan"),  # NaN bounds must be skipped
        }
        with mlflow.start_run() as run:
            log_ci_artifact(metrics, split="test")
            run_id = run.info.run_id

        client = MlflowClient()
        local_path = client.download_artifacts(run_id, "metrics/test_ci.json", str(tmp_path))
        data = json.loads(open(local_path).read())
        assert data == {"F1": {"mean": 0.876, "lower": 0.851, "upper": 0.901}}

    def test_no_op_when_no_ci_data(self, mlflow_tmp_tracking):
        with mlflow.start_run() as run:
            log_ci_artifact({"F1": "n/a"}, split="test")
            run_id = run.info.run_id

        assert MlflowClient().list_artifacts(run_id, "metrics") == []


# ---------------------------------------------------------------------------
# log_dataset_info
# ---------------------------------------------------------------------------


class TestLogDatasetInfo:
    def test_logs_per_split_counts_and_entity_counts(self, mlflow_tmp_tracking, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame(
            {
                "split": ["train", "train", "train", "val", "test"],
                "label": [1, 0, 0, 1, 0],
                "patient_id": ["p1", "p1", "p2", "p3", "p1"],
                "video_id": ["v1", "v1", "v2", "v3", "v1"],
            }
        ).to_csv(manifest_path, index=False)

        with mlflow.start_run() as run:
            log_dataset_info(manifest_path)
            run_id = run.info.run_id

        params = MlflowClient().get_run(run_id).data.params
        assert params["data_train_n"] == "3"
        assert params["data_train_n_positive"] == "1"
        assert params["data_train_pos_ratio"] == "0.3333"
        assert params["data_val_n"] == "1"
        assert params["data_test_n"] == "1"
        assert params["data_n_patients"] == "3"
        assert params["data_n_videos"] == "3"
        assert params["data_total_frames"] == "5"

    def test_missing_optional_columns_are_skipped(self, mlflow_tmp_tracking, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame({"split": ["train"], "label": [1]}).to_csv(manifest_path, index=False)

        with mlflow.start_run() as run:
            log_dataset_info(manifest_path)  # must not raise without patient_id/video_id
            run_id = run.info.run_id

        params = MlflowClient().get_run(run_id).data.params
        assert "data_n_patients" not in params
        assert params["data_total_frames"] == "1"


# ---------------------------------------------------------------------------
# log_figures / log_figures_from_dir / log_confusion_matrix
# ---------------------------------------------------------------------------


class TestLogFigures:
    def test_logs_one_png_per_figure_under_subdir(self, mlflow_tmp_tracking, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        with mlflow.start_run() as run:
            log_figures({"roc_curve": fig}, subdir="test")
            run_id = run.info.run_id

        client = MlflowClient()
        names = [a.path for a in client.list_artifacts(run_id, "test")]
        assert names == ["test/roc_curve.png"]

    def test_closes_figure_after_logging(self, mlflow_tmp_tracking):
        fig, ax = plt.subplots()
        with mlflow.start_run():
            log_figures({"f": fig})
        assert not plt.fignum_exists(fig.number)


class TestLogFiguresFromDir:
    def test_logs_all_pngs_in_directory(self, mlflow_tmp_tracking, tmp_path):
        fig_dir = tmp_path / "figs"
        fig_dir.mkdir()
        (fig_dir / "a.png").write_bytes(b"fake-png-a")
        (fig_dir / "b.png").write_bytes(b"fake-png-b")
        (fig_dir / "c.txt").write_text("not a figure")

        with mlflow.start_run() as run:
            log_figures_from_dir(fig_dir, subdir="explainability")
            run_id = run.info.run_id

        names = sorted(a.path for a in MlflowClient().list_artifacts(run_id, "explainability"))
        assert names == ["explainability/a.png", "explainability/b.png"]


class TestLogConfusionMatrix:
    def test_logs_confusion_matrix_png(self, mlflow_tmp_tracking):
        cm = np.array([[8, 2], [1, 9]])
        with mlflow.start_run() as run:
            log_confusion_matrix(cm, threshold=0.5, model_name="vits16", prefix="test")
            run_id = run.info.run_id

        names = [a.path for a in MlflowClient().list_artifacts(run_id, "test")]
        assert names == ["test/vits16_confusion_matrix.png"]


# ---------------------------------------------------------------------------
# get_best_run
# ---------------------------------------------------------------------------


class TestGetBestRun:
    def test_returns_highest_metric_run(self, mlflow_tmp_tracking):
        with mlflow.start_run():
            mlflow.log_metric("test__f1_mean", 0.70)
        with mlflow.start_run() as best:
            mlflow.log_metric("test__f1_mean", 0.90)
            best_id = best.info.run_id

        result = get_best_run(mlflow_tmp_tracking, "test__f1_mean")
        assert result["run_id"] == best_id
        assert result["metrics"]["test__f1_mean"] == 0.90

    def test_returns_none_for_unknown_experiment(self, mlflow_tmp_tracking):
        assert get_best_run("no_such_experiment", "test__f1_mean") is None

    def test_returns_none_when_no_runs_have_metric(self, mlflow_tmp_tracking):
        with mlflow.start_run():
            mlflow.log_param("model", "x")  # no metric logged
        assert get_best_run(mlflow_tmp_tracking, "test__f1_mean") is None


# ---------------------------------------------------------------------------
# compare_runs_to_markdown
# ---------------------------------------------------------------------------


class TestCompareRunsToMarkdown:
    def test_excludes_nested_runs_and_builds_table(self, mlflow_tmp_tracking, tmp_path):
        with mlflow.start_run(run_name="parent") as parent:
            mlflow.log_param("model", "vits16_gastronet")
            mlflow.log_metric("cv_mean_val_auroc", 0.91)
            with mlflow.start_run(run_name="fold_1", nested=True):
                mlflow.log_metric("cv_mean_val_auroc", 0.99)  # must not leak into the table

        save_path = tmp_path / "runs_comparison.md"
        md = compare_runs_to_markdown(
            experiment_name=mlflow_tmp_tracking,
            metrics=["cv_mean_val_auroc"],
            save_path=save_path,
        )

        assert "| run | model | cv_mean_val_auroc |" in md
        assert "0.9100" in md
        assert "0.9900" not in md  # the nested fold run's metric is excluded
        assert save_path.read_text(encoding="utf-8") == md

    def test_unknown_experiment_returns_message(self, mlflow_tmp_tracking):
        result = compare_runs_to_markdown("no_such_experiment", ["f1"])
        assert "not found" in result

    def test_no_runs_returns_message(self, mlflow_tmp_tracking):
        assert compare_runs_to_markdown(mlflow_tmp_tracking, ["f1"]) == "No top-level runs found."


# ---------------------------------------------------------------------------
# register_best_model / promote_model
# ---------------------------------------------------------------------------


class TestRegisterAndPromoteModel:
    def test_register_creates_model_version(self, mlflow_tmp_tracking):
        with mlflow.start_run() as run:
            run_id = run.info.run_id

        version = register_best_model(run_id, "test_registered_model", description="F1=0.90")
        assert version is not None

        client = MlflowClient()
        mv = client.get_model_version("test_registered_model", version)
        assert mv.run_id == run_id

    def test_promote_sets_champion_alias(self, mlflow_tmp_tracking):
        with mlflow.start_run() as run:
            run_id = run.info.run_id
        version = register_best_model(run_id, "test_promote_model")

        promote_model("test_promote_model", version, alias="champion")

        client = MlflowClient()
        champion = client.get_model_version_by_alias("test_promote_model", "champion")
        assert champion.version == version

    def test_register_twice_reuses_existing_registered_model(self, mlflow_tmp_tracking):
        with mlflow.start_run() as run1:
            run_id1 = run1.info.run_id
        with mlflow.start_run() as run2:
            run_id2 = run2.info.run_id

        v1 = register_best_model(run_id1, "shared_model")
        v2 = register_best_model(run_id2, "shared_model")
        assert v1 != v2  # two distinct versions of the same registered model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
