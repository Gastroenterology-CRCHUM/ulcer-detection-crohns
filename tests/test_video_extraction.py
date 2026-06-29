"""Tests for src/data/video_extraction."""

from __future__ import annotations

import pytest

from src.data.video_extraction import (
    VIDEO_EXTS,
    build_video_index_generic,
    collect_frames_from_dir,
    sample_timestamps,
)

# ---------------------------------------------------------------------------
# sample_timestamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fps, expected",
    [
        (1.0, [0.0, 1.0, 2.0]),
        (0.5, [0.0, 2.0]),
    ],
)
def test_sample_timestamps_fps(fps, expected):
    assert sample_timestamps(0.0, 3.0, fps_target=fps) == pytest.approx(expected)


def test_sample_timestamps_empty_when_end_before_start():
    assert sample_timestamps(5.0, 3.0, fps_target=1.0) == []


def test_sample_timestamps_empty_when_zero_fps():
    assert sample_timestamps(0.0, 1.0, fps_target=0.0) == []


def test_sample_timestamps_empty_when_equal():
    assert sample_timestamps(2.0, 2.0, fps_target=1.0) == []


def test_sample_timestamps_start_offset():
    ts = sample_timestamps(10.0, 12.0, fps_target=1.0)
    assert ts == pytest.approx([10.0, 11.0])


# ---------------------------------------------------------------------------
# build_video_index_generic
# ---------------------------------------------------------------------------


def test_build_video_index_ignores_non_video(tmp_path):
    (tmp_path / "notes.txt").touch()
    (tmp_path / "image.jpg").touch()
    idx = build_video_index_generic(tmp_path)
    assert "notes" not in idx
    assert "image" not in idx


def test_build_video_index_is_lowercase(tmp_path):
    (tmp_path / "VideoName.MP4").touch()
    idx = build_video_index_generic(tmp_path)
    assert "videoname" in idx


def test_build_video_index_recurses_subdirs(tmp_path):
    sub = tmp_path / "session_01"
    sub.mkdir()
    (sub / "deep.mp4").touch()
    idx = build_video_index_generic(tmp_path)
    assert "deep" in idx


def test_build_video_index_all_known_exts(tmp_path):
    for i, ext in enumerate(sorted(VIDEO_EXTS)):
        (tmp_path / f"vid_{i}{ext}").touch()
    idx = build_video_index_generic(tmp_path)
    assert len(idx) == len(VIDEO_EXTS)


# ---------------------------------------------------------------------------
# collect_frames_from_dir
# ---------------------------------------------------------------------------


def test_collect_frames_finds_jpg_png_bmp(tmp_path):
    for name in ("a.jpg", "b.png", "c.bmp"):
        (tmp_path / name).touch()
    frames = collect_frames_from_dir(tmp_path)
    assert {f.name for f in frames} == {"a.jpg", "b.png", "c.bmp"}


def test_collect_frames_ignores_non_image(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.txt").touch()
    frames = collect_frames_from_dir(tmp_path)
    assert len(frames) == 1


def test_collect_frames_is_sorted(tmp_path):
    for name in ("c.jpg", "a.jpg", "b.jpg"):
        (tmp_path / name).touch()
    frames = collect_frames_from_dir(tmp_path)
    assert [f.name for f in frames] == ["a.jpg", "b.jpg", "c.jpg"]


def test_collect_frames_case_insensitive_ext(tmp_path):
    (tmp_path / "frame.JPG").touch()
    (tmp_path / "frame2.JPEG").touch()
    frames = collect_frames_from_dir(tmp_path)
    assert len(frames) == 2


def test_collect_frames_recurses_subdirs(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.jpg").touch()
    frames = collect_frames_from_dir(tmp_path)
    assert any(f.name == "deep.jpg" for f in frames)


def test_collect_frames_empty_dir(tmp_path):
    assert collect_frames_from_dir(tmp_path) == []
