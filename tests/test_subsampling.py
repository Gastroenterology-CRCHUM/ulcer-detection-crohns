"""Tests for src/data/subsampling (backbone-free paths only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.subsampling import visual_subsample


def _dummy_paths(n: int, tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i:04d}.jpg"
        p.touch()
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# visual_subsample — no backbone (uniform stride)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 10])
def test_subsample_no_op_at_or_under_limit(n, tmp_path):
    paths = _dummy_paths(n, tmp_path)
    assert visual_subsample(paths, max_count=10) == paths


def test_subsample_uniform_stride_returns_correct_count(tmp_path):
    paths = _dummy_paths(100, tmp_path)
    result = visual_subsample(paths, max_count=10, backbone=None)
    assert len(result) == 10


def test_subsample_uniform_stride_selects_from_input(tmp_path):
    paths = _dummy_paths(50, tmp_path)
    result = visual_subsample(paths, max_count=5, backbone=None)
    assert all(p in paths for p in result)


def test_subsample_preserves_path_order(tmp_path):
    paths = _dummy_paths(50, tmp_path)
    result = visual_subsample(paths, max_count=5, backbone=None)
    indices = [paths.index(p) for p in result]
    assert indices == sorted(indices)


def test_subsample_returns_list(tmp_path):
    paths = _dummy_paths(20, tmp_path)
    assert isinstance(visual_subsample(paths, max_count=5, backbone=None), list)


def test_subsample_max_count_1(tmp_path):
    paths = _dummy_paths(20, tmp_path)
    result = visual_subsample(paths, max_count=1, backbone=None)
    assert len(result) == 1


def test_subsample_empty_input():
    assert visual_subsample([], max_count=5) == []
