"""Tests for src/evaluation/runner.py — focused on run_delong."""

import numpy as np
import pytest

from src.evaluation.runner import run_delong


def _make_results(n: int = 60, seed: int = 0) -> dict[str, dict]:
    """Produce a fake evaluate_all_models-style results dict for two models."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    video_ids = [f"clip_{i // 3}" for i in range(n)]

    return {
        "ModelA": {
            "labels": labels,
            "probs": np.clip(labels + rng.normal(0, 0.3, n), 0, 1).astype(float),
            "preds": (labels + rng.integers(-1, 2, n)).clip(0, 1).astype(int),
            "video_ids": video_ids,
        },
        "ModelB": {
            "labels": labels,
            "probs": np.clip(rng.uniform(0, 1, n), 0, 1).astype(float),
            "preds": rng.integers(0, 2, n).astype(int),
            "video_ids": video_ids,
        },
    }


class TestRunDelong:
    def test_frame_level_returns_tuple(self):
        results = _make_results()
        p_matrix, df, fig = run_delong(results, level="frame")
        assert p_matrix.shape == (2, 2)
        assert len(df) == 1  # one pair for two models
        assert fig is not None

    def test_frame_level_df_columns(self):
        results = _make_results()
        _, df, _ = run_delong(results, level="frame")
        required = {"Model A", "Model B", "AUC A", "AUC B", "ΔAUC", "z", "p-value", "significant"}
        assert required.issubset(df.columns)

    def test_clip_level_returns_tuple(self):
        results = _make_results()
        p_matrix, df, fig = run_delong(results, level="clip")
        assert p_matrix.shape == (2, 2)
        assert len(df) == 1
        assert fig is not None

    def test_invalid_level_raises(self):
        results = _make_results()
        with pytest.raises(ValueError, match="level must be"):
            run_delong(results, level="patient")

    def test_save_csv(self, tmp_path):
        results = _make_results()
        csv_path = str(tmp_path / "delong.csv")
        _, df, _ = run_delong(results, level="frame", save_csv=csv_path)
        import pandas as pd

        loaded = pd.read_csv(csv_path)
        assert list(loaded.columns) == list(df.columns)

    def test_custom_alpha(self):
        results = _make_results()
        _, df_strict, _ = run_delong(results, level="frame", alpha=1e-10)
        _, df_loose, _ = run_delong(results, level="frame", alpha=0.99)
        assert df_loose["significant"].sum() >= df_strict["significant"].sum()

    def test_three_models_three_pairs(self):
        rng = np.random.default_rng(42)
        n = 60
        labels = rng.integers(0, 2, n)
        video_ids = [f"clip_{i // 3}" for i in range(n)]
        results = {
            f"Model{k}": {
                "labels": labels,
                "probs": np.clip(rng.uniform(0, 1, n), 0, 1).astype(float),
                "preds": rng.integers(0, 2, n).astype(int),
                "video_ids": video_ids,
            }
            for k in ("A", "B", "C")
        }
        _, df, _ = run_delong(results, level="frame")
        assert len(df) == 3  # C(3, 2) = 3 pairs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
