"""
tests/test_training_trainer.py
==============================
Unit tests for checkpoint loading helpers.
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.training.trainer import load_best_checkpoint


class DummyModel:
    def __init__(self, name: str):
        self.name = name
        self.base_model = MagicMock()


class TestLoadBestCheckpoint:
    """Test load_best_checkpoint function."""

    def test_loads_latest_checkpoint(self, checkpoint_workspace, monkeypatch):
        model_name = "dummy_model"
        base_dir = checkpoint_workspace / "output" / "models" / model_name
        old_dir = base_dir / "20250101_120000"
        new_dir = base_dir / "20250101_130000"
        old_dir.mkdir(parents=True)
        new_dir.mkdir(parents=True)

        old_state = {"weight": torch.tensor([1.0])}
        new_state = {"weight": torch.tensor([2.0])}
        torch.save(old_state, old_dir / "best.pt")
        torch.save(new_state, new_dir / "best.pt")

        # Force deterministic mtime ordering so the test isn't flaky on
        # file systems with 1-second timestamp resolution.
        t_old = time.time() - 10
        t_new = time.time()
        os.utime(old_dir / "best.pt", (t_old, t_old))
        os.utime(new_dir / "best.pt", (t_new, t_new))

        monkeypatch.chdir(checkpoint_workspace)
        model = DummyModel(model_name)

        checkpoint_dir = load_best_checkpoint(model, device=torch.device("cpu"))

        assert checkpoint_dir == Path("output") / "models" / model_name / "20250101_130000"
        model.base_model.load_state_dict.assert_called_once()

    def test_missing_checkpoint_raises(self, checkpoint_workspace, monkeypatch):
        monkeypatch.chdir(checkpoint_workspace)
        model = DummyModel("missing_model")

        with pytest.raises(FileNotFoundError):
            load_best_checkpoint(model, device=torch.device("cpu"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
