"""Tests for src/data/subsampling.py."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.data.subsampling import visual_subsample  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frames(tmp_path: Path, n: int) -> list[Path]:
    """Create n small JPEG images and return their paths."""
    paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i:04d}.jpg"
        Image.new("RGB", (32, 32), color=(i * 10 % 256, 0, 0)).save(p)
        paths.append(p)
    return paths


def _mock_backbone(embed_dim: int = 8):
    """Return a backbone mock that produces deterministic embeddings."""
    backbone = MagicMock()

    def _forward(batch):
        # Return a tensor with shape (batch, embed_dim) filled with batch index
        import torch

        n = batch.shape[0]
        # Spread embeddings so farthest-point sampling is deterministic
        out = torch.zeros(n, embed_dim)
        for i in range(n):
            out[i, i % embed_dim] = float(i + 1)
        return out

    backbone.eval.return_value = backbone
    backbone.__call__ = lambda self_, x: _forward(x)
    backbone.return_value = _forward(MagicMock())
    # Make the backbone callable
    backbone.side_effect = _forward
    return backbone


# ---------------------------------------------------------------------------
# visual_subsample — no backbone
# ---------------------------------------------------------------------------


class TestVisualSubsampleNoBackbone:
    def test_returns_all_when_under_limit(self, tmp_path):
        frames = _make_frames(tmp_path, 5)
        result = visual_subsample(frames, max_count=10, backbone=None)
        assert result == frames

    def test_returns_all_when_equal_to_limit(self, tmp_path):
        frames = _make_frames(tmp_path, 5)
        result = visual_subsample(frames, max_count=5, backbone=None)
        assert result == frames

    def test_uniform_stride_count(self, tmp_path):
        frames = _make_frames(tmp_path, 20)
        result = visual_subsample(frames, max_count=5, backbone=None)
        assert len(result) == 5

    def test_uniform_stride_returns_paths(self, tmp_path):
        frames = _make_frames(tmp_path, 10)
        result = visual_subsample(frames, max_count=3, backbone=None)
        assert all(isinstance(p, Path) for p in result)
        assert all(p in frames for p in result)

    def test_uniform_stride_includes_first_and_last(self, tmp_path):
        frames = _make_frames(tmp_path, 20)
        result = visual_subsample(frames, max_count=5, backbone=None)
        assert frames[0] in result
        assert frames[-1] in result

    def test_max_count_one(self, tmp_path):
        frames = _make_frames(tmp_path, 10)
        result = visual_subsample(frames, max_count=1, backbone=None)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# visual_subsample — with mock backbone (farthest-point sampling)
# ---------------------------------------------------------------------------


class TestVisualSubsampleWithBackbone:
    def test_correct_count_returned(self, tmp_path):
        frames = _make_frames(tmp_path, 10)
        backbone = _mock_backbone(embed_dim=16)
        result = visual_subsample(frames, max_count=4, backbone=backbone)
        assert len(result) == 4

    def test_result_is_subset_of_input(self, tmp_path):
        frames = _make_frames(tmp_path, 8)
        backbone = _mock_backbone(embed_dim=16)
        result = visual_subsample(frames, max_count=3, backbone=backbone)
        assert all(p in frames for p in result)

    def test_result_is_sorted(self, tmp_path):
        """Farthest-point sampling returns indices in sorted order."""
        frames = _make_frames(tmp_path, 8)
        backbone = _mock_backbone(embed_dim=16)
        result = visual_subsample(frames, max_count=3, backbone=backbone)
        assert result == sorted(result, key=lambda p: frames.index(p))

    def test_no_duplicates(self, tmp_path):
        frames = _make_frames(tmp_path, 8)
        backbone = _mock_backbone(embed_dim=16)
        result = visual_subsample(frames, max_count=4, backbone=backbone)
        assert len(result) == len(set(result))

    def test_returns_all_when_under_limit_with_backbone(self, tmp_path):
        """Even with backbone, returns all frames if n <= max_count."""
        frames = _make_frames(tmp_path, 3)
        backbone = _mock_backbone(embed_dim=16)
        result = visual_subsample(frames, max_count=10, backbone=backbone)
        assert result == frames


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
