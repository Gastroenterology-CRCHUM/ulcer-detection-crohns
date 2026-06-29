"""Tests for src/data/annotation_loaders."""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.data.annotation_loaders import _hms_to_seconds, load_ulcer_annotations

# ---------------------------------------------------------------------------
# _hms_to_seconds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("00:01:30", 90.0),
        ("01:00:00", 3600.0),
        ("00:00:00", 0.0),
        ("00:00:00.5", 0.5),
    ],
)
def test_hms_to_seconds_string(value, expected):
    assert _hms_to_seconds(value) == pytest.approx(expected)


def test_hms_to_seconds_timedelta():
    td = datetime.timedelta(hours=1, minutes=2, seconds=3)
    assert _hms_to_seconds(td) == pytest.approx(3723.0)


def test_hms_to_seconds_nan_returns_none():
    import math

    assert _hms_to_seconds(float("nan")) is None
    assert _hms_to_seconds(None) is None
    assert _hms_to_seconds(math.nan) is None


def test_hms_to_seconds_invalid_returns_none():
    assert _hms_to_seconds("not_a_time") is None
    assert _hms_to_seconds("12:34") is None  # missing seconds field


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_ulcer_excel(tmp_path: Path) -> Path:
    path = tmp_path / "timestamps.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "record_id": ["vid_01", "vid_01"],
                "start_time": ["00:01:00", "00:03:00"],
                "end_time": ["00:02:00", "00:04:00"],
                "sample_number": [1, 2],
                "Size:": ["small", "large"],
            }
        ).to_excel(writer, sheet_name="Ulcer timestamps", index=False)
        pd.DataFrame(
            {
                "record_id": ["vid_02"],
                "start_time": ["00:05:00"],
                "end_time": ["00:06:00"],
                "sample_number": [1],
                "Size:": [None],
            }
        ).to_excel(writer, sheet_name="Non-Ulcer timestamps", index=False)
    return path


# ---------------------------------------------------------------------------
# load_ulcer_annotations
# ---------------------------------------------------------------------------


def test_load_ulcer_annotations_row_count(tmp_path):
    df = load_ulcer_annotations(_make_ulcer_excel(tmp_path))
    assert len(df) == 3


def test_load_ulcer_annotations_required_columns(tmp_path):
    df = load_ulcer_annotations(_make_ulcer_excel(tmp_path))
    assert {"record_id", "start_s", "end_s", "label"}.issubset(df.columns)


def test_load_ulcer_annotations_labels(tmp_path):
    df = load_ulcer_annotations(_make_ulcer_excel(tmp_path))
    assert set(df["label"].unique()) == {0, 1}


def test_load_ulcer_annotations_label_assignment(tmp_path):
    df = load_ulcer_annotations(_make_ulcer_excel(tmp_path))
    assert (df[df["record_id"] == "vid_01"]["label"] == 1).all()
    assert (df[df["record_id"] == "vid_02"]["label"] == 0).all()


def test_load_ulcer_annotations_seconds_conversion(tmp_path):
    df = load_ulcer_annotations(_make_ulcer_excel(tmp_path))
    row = df[df["record_id"] == "vid_01"].iloc[0]
    assert row["start_s"] == pytest.approx(60.0)
    assert row["end_s"] == pytest.approx(120.0)


def test_load_ulcer_annotations_end_after_start(tmp_path):
    df = load_ulcer_annotations(_make_ulcer_excel(tmp_path))
    assert (df["end_s"] > df["start_s"]).all()


def test_load_ulcer_annotations_missing_sheet_raises(tmp_path):
    path = tmp_path / "bad.xlsx"
    pd.DataFrame({"x": [1]}).to_excel(path, sheet_name="Wrong Sheet", index=False)
    with pytest.raises(ValueError, match="Cannot read sheet"):
        load_ulcer_annotations(path)
