"""Tests for scripts/ulcer/create_manifest.py and scripts/ulcer/eda.py."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.ulcer.create_manifest import (
    DatasetPreparer,
    _extract_frame_number,
    _parse_segment_number,
    build_parser,
    load_ulcer_size_lookup,
)
from scripts.ulcer.eda import DatasetEDA

# ---------------------------------------------------------------------------
# _parse_segment_number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "segment_id, expected",
    [
        ("ulcer_1", 1),
        ("normal_12", 12),
        ("segment_0", 0),
        ("no_underscore_number_42", 42),
    ],
)
def test_parse_segment_number(segment_id, expected):
    assert _parse_segment_number(segment_id) == expected


def test_parse_segment_number_bad_input():
    assert _parse_segment_number("noint") == -1
    assert _parse_segment_number("") == -1


# ---------------------------------------------------------------------------
# _extract_frame_number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("frame_001.jpg", 1),
        ("frame_042.png", 42),
        ("vid_01__ulcer_1__frame_100.jpg", 100),
    ],
)
def test_extract_frame_number(filename, expected):
    assert _extract_frame_number(filename) == expected


def test_extract_frame_number_bad_input():
    assert _extract_frame_number("noints.jpg") == -1


# ---------------------------------------------------------------------------
# load_ulcer_size_lookup
# ---------------------------------------------------------------------------


def test_load_ulcer_size_lookup_missing_file(tmp_path):
    result = load_ulcer_size_lookup(str(tmp_path / "missing.xlsx"), "Sheet1")
    assert result == {}


# ---------------------------------------------------------------------------
# DatasetPreparer.scan_directory
# ---------------------------------------------------------------------------


def _build_dir_tree(root: Path, n_ulcer_frames=3, n_nonulcer_frames=2) -> None:
    for class_name, count in (("Ulcer", n_ulcer_frames), ("NonUlcer", n_nonulcer_frames)):
        seg = root / class_name / "vid_01" / f"{class_name.lower()}_1"
        seg.mkdir(parents=True)
        for i in range(count):
            (seg / f"frame_{i:03d}.jpg").write_bytes(b"x")


def test_scan_directory(tmp_path):
    _build_dir_tree(tmp_path)
    preparer = DatasetPreparer()
    df = preparer.scan_directory(tmp_path, size_lookup={})
    assert len(df) == 5
    assert set(df["class_name"].unique()) == {"Ulcer", "NonUlcer"}
    assert set(df["video_id"].unique()) == {"vid_01"}
    assert set(df["label"].unique()) == {0, 1}
    assert df[df["class_name"] == "Ulcer"]["label"].unique()[0] == 1
    assert df[df["class_name"] == "NonUlcer"]["label"].unique()[0] == 0


def test_scan_directory_missing_class(tmp_path, caplog):
    (tmp_path / "Ulcer" / "vid_01" / "ulcer_1").mkdir(parents=True)
    (tmp_path / "Ulcer" / "vid_01" / "ulcer_1" / "frame_000.jpg").write_bytes(b"x")
    preparer = DatasetPreparer()
    df = preparer.scan_directory(tmp_path, size_lookup={})
    assert len(df) == 1
    assert (df["label"] == 1).all()


# ---------------------------------------------------------------------------
# DatasetPreparer.create_patient_level_splits
# ---------------------------------------------------------------------------


def _make_df_for_splits(n_patients=6) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        pid = f"vid_{i:02d}"
        label = i % 2
        rows.append(
            {
                "patient_id": pid,
                "video_id": pid,
                "label": label,
                "class_name": "Ulcer" if label else "NonUlcer",
                "segment_id": "seg_1",
                "segment_number": 1,
                "frame_number": 0,
                "image_path": f"data/{pid}/seg_1/frame_000.jpg",
                "relative_path": f"{pid}/seg_1/frame_000.jpg",
                "clip_key": f"{pid}__seg_1",
                "ulcer_size": None,
            }
        )
    return pd.DataFrame(rows)


def test_create_patient_level_splits_assigns_all(tmp_path):
    df = _make_df_for_splits(12)
    preparer = DatasetPreparer()
    # Populate patient_info from df so the splitter knows about all patients
    n_patients = df["patient_id"].nunique()
    for pid in df["patient_id"].unique():
        sub = df[df["patient_id"] == pid]
        preparer.patient_info[pid] = {
            "video_id": pid,
            "has_ulcer": bool(sub["label"].max()),
            "has_non_ulcer": bool(1 - sub["label"].max()),
            "ulcer_frames": int((sub["label"] == 1).sum()),
            "non_ulcer_frames": int((sub["label"] == 0).sum()),
            "total_frames": len(sub),
            "segments": ["seg_1"],
            "ulcer_presence": float(sub["label"].mean()),
        }
    df, split_info = preparer.create_patient_level_splits(df)
    assert df["split"].notna().all()
    assert set(df["split"].unique()).issubset({"train", "val", "test"})
    total_patients = (
        split_info["splits"]["train"]["n_patients"]
        + split_info["splits"]["val"]["n_patients"]
        + split_info["splits"]["test"]["n_patients"]
    )
    assert total_patients == n_patients


# ---------------------------------------------------------------------------
# build_parser (ulcer create_manifest)
# ---------------------------------------------------------------------------


def _fake_paths() -> SimpleNamespace:
    return SimpleNamespace(
        ulcer_filtrated_dir=Path("data/ulcer/filtrated"),
        ulcer_splits_dir=Path("data/ulcer/splits"),
        ulcer_raw_dir=Path("data/ulcer/raw"),
    )


def test_build_parser_defaults(monkeypatch):
    import scripts.ulcer.create_manifest as cm

    monkeypatch.setattr(cm, "get_default_paths", _fake_paths)
    parser = build_parser()
    args = parser.parse_args([])
    assert Path(args.input_dir).as_posix() == "data/ulcer/filtrated"
    assert Path(args.splits_dir).as_posix() == "data/ulcer/splits"
    assert args.train_ratio == 0.70
    assert args.val_ratio == 0.15
    assert args.test_ratio == 0.15
    assert args.seed == 42


# ---------------------------------------------------------------------------
# DatasetEDA — init and load_data
# ---------------------------------------------------------------------------


def test_dataset_eda_creates_output_dir(tmp_path):
    DatasetEDA(splits_dir=str(tmp_path / "splits"), output_dir=str(tmp_path / "eda"))
    assert (tmp_path / "eda").exists()


def test_dataset_eda_load_data_missing_manifest(tmp_path):
    eda = DatasetEDA(splits_dir=str(tmp_path / "splits"), output_dir=str(tmp_path / "eda"))
    with pytest.raises(FileNotFoundError):
        eda.load_data()


def test_dataset_eda_load_and_compute(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    rows = []
    for i in range(4):
        pid = f"vid_{i:02d}"
        rows.append(
            {
                "image_path": f"{pid}/seg_1/f.jpg",
                "video_id": pid,
                "patient_id": pid,
                "class_name": "Ulcer" if i % 2 else "NonUlcer",
                "label": i % 2,
                "segment_id": "seg_1",
                "segment_number": 1,
                "frame_number": 0,
                "clip_key": f"{pid}__seg_1",
                "relative_path": f"{pid}/seg_1/f.jpg",
                "ulcer_size": None,
                "split": ["train", "train", "val", "test"][i],
            }
        )
    pd.DataFrame(rows).to_csv(splits / "dataset_manifest.csv", index=False)
    eda = DatasetEDA(splits_dir=str(splits), output_dir=str(tmp_path / "eda"))
    eda.load_data()
    assert eda.df is not None and len(eda.df) == 4
    stats = eda.compute_dataset_statistics()
    assert stats["frames"]["total"] == 4
    assert stats["patients"]["total"] == 4
