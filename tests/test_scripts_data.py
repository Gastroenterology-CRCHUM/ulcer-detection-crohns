"""Tests for scripts/data/eda_utils.py pure-logic functions."""

import numpy as np
import pandas as pd

from src.data.eda_utils import (
    count_images,
    entity_summaries,
    manifest_quality_checks,
    split_diagnostics,
    to_serializable,
)

# ---------------------------------------------------------------------------
# to_serializable
# ---------------------------------------------------------------------------


def test_to_serializable_numpy_int():
    assert to_serializable(np.int64(7)) == 7
    assert isinstance(to_serializable(np.int32(3)), int)


def test_to_serializable_numpy_float():
    result = to_serializable(np.float32(1.5))
    assert isinstance(result, float)
    assert abs(result - 1.5) < 1e-4


def test_to_serializable_nested():
    obj = {"a": np.int64(1), "b": [np.float64(2.0), {"c": np.int32(3)}]}
    result = to_serializable(obj)
    assert result == {"a": 1, "b": [2.0, {"c": 3}]}
    assert isinstance(result["a"], int)


def test_to_serializable_passthrough():
    assert to_serializable("hello") == "hello"
    assert to_serializable(42) == 42
    assert to_serializable(None) is None


# ---------------------------------------------------------------------------
# count_images
# ---------------------------------------------------------------------------


def test_count_images_missing_dir(tmp_path):
    assert count_images(tmp_path / "nonexistent") == 0


def test_count_images_counts_only_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpeg").write_bytes(b"x")
    assert count_images(tmp_path) == 3


# ---------------------------------------------------------------------------
# manifest_quality_checks
# ---------------------------------------------------------------------------


def _make_manifest(n=4):
    return pd.DataFrame(
        {
            "relative_path": [f"p{i}.jpg" for i in range(n)],
            "label": [0, 1, 0, 1][:n],
            "video_id": [f"vid_{i}" for i in range(n)],
            "patient_id": [f"p{i}" for i in range(n)],
            "segment_id": [f"seg_{i}" for i in range(n)],
            "frame_number": list(range(n)),
            "clip_key": [f"vid_{i}__seg_{i}" for i in range(n)],
            "split": ["train", "train", "val", "test"][:n],
        }
    )


def test_manifest_quality_checks_empty():
    result = manifest_quality_checks(pd.DataFrame())
    assert result["rows"] == 0
    assert result["duplicate_relative_path"] == 0


def test_manifest_quality_checks_no_duplicates():
    df = _make_manifest()
    result = manifest_quality_checks(df)
    assert result["rows"] == 4
    assert result["duplicate_relative_path"] == 0
    assert result["duplicate_clip_frame"] == 0


def test_manifest_quality_checks_detects_duplicate_path():
    df = _make_manifest()
    df.loc[3, "relative_path"] = df.loc[0, "relative_path"]
    result = manifest_quality_checks(df)
    assert result["duplicate_relative_path"] == 1


# ---------------------------------------------------------------------------
# split_diagnostics
# ---------------------------------------------------------------------------


def test_split_diagnostics_empty():
    result = split_diagnostics(pd.DataFrame())
    assert result["split_counts"] == {}


def test_split_diagnostics_no_split_column():
    df = pd.DataFrame({"label": [0, 1]})
    result = split_diagnostics(df)
    assert result["split_counts"] == {}


def test_split_diagnostics_counts():
    df = _make_manifest()
    result = split_diagnostics(df)
    assert result["split_counts"]["train"] == 2
    assert result["split_counts"]["val"] == 1
    assert result["split_counts"]["test"] == 1


def test_split_diagnostics_no_patient_leakage():
    df = _make_manifest()
    result = split_diagnostics(df)
    for pair_key, info in result["patient_leakage"].items():
        assert info["count"] == 0, f"Unexpected leakage in {pair_key}"


def test_split_diagnostics_detects_leakage():
    df = _make_manifest()
    df.loc[2, "patient_id"] = df.loc[0, "patient_id"]  # same patient in train and val
    result = split_diagnostics(df)
    assert result["patient_leakage"]["train_vs_val"]["count"] == 1


# ---------------------------------------------------------------------------
# entity_summaries
# ---------------------------------------------------------------------------


def test_entity_summaries_empty():
    summary, vid_df, clip_df = entity_summaries(pd.DataFrame())
    assert summary["videos"] == 0
    assert summary["patients"] == 0
    assert vid_df.empty
    assert clip_df.empty


def test_entity_summaries_counts():
    df = _make_manifest()
    summary, vid_df, clip_df = entity_summaries(df)
    assert summary["videos"] == 4
    assert summary["patients"] == 4
    assert summary["clips"] == 4
    assert not vid_df.empty
    assert not clip_df.empty


def test_entity_summaries_frames_per_split():
    df = _make_manifest()
    summary, _, _ = entity_summaries(df)
    assert summary["frames_per_split"]["train"] == 2
