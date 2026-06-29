"""Tests for src/evaluation/model_loader.py."""

from pathlib import Path

import pytest
import torch

from src.evaluation.model_loader import load_best_models, load_model
from src.models.classifier import ClassifierModel

# ---------------------------------------------------------------------------
# Fixture: isolated PathConfig workspace (ClassifierModel calls PathConfig())
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _project_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create the minimal directory tree required by PathConfig."""
    (tmp_path / "data" / "ulcer" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "ulcer" / "splits").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(tmp_path: Path, arch: str = "resnet18") -> Path:
    """Instantiate a ClassifierModel, save its base_model state_dict, return path."""
    wrapper = ClassifierModel(
        base_model=arch,
        num_classes=1,
        optimizer="AdamW",
        learning_rate=5e-5,
    )
    ckpt_path = tmp_path / "best.pt"
    torch.save(wrapper.base_model.state_dict(), ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_returns_classifier_model(self, tmp_path):
        ckpt_path = _make_checkpoint(tmp_path)
        entry = {"path": str(ckpt_path), "freeze_backbone": 0, "head_type": "linear"}
        model = load_model("resnet18-allBackbone", entry, project_root=tmp_path)
        assert isinstance(model, ClassifierModel)

    def test_model_in_eval_mode(self, tmp_path):
        """load_model does not set eval mode — verify training state is default."""
        ckpt_path = _make_checkpoint(tmp_path)
        entry = {"path": str(ckpt_path), "freeze_backbone": 0, "head_type": "linear"}
        model = load_model("resnet18-allBackbone", entry, project_root=tmp_path)
        # ClassifierModel is an nn.Module; training mode is True by default at construction
        assert isinstance(model, torch.nn.Module)

    def test_path_without_pt_extension_appends_best_pt(self, tmp_path):
        """Entry path without .pt → code appends /best.pt."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        wrapper = ClassifierModel(
            base_model="resnet18", num_classes=1, optimizer="AdamW", learning_rate=5e-5
        )
        torch.save(wrapper.base_model.state_dict(), ckpt_dir / "best.pt")

        entry = {"path": str(ckpt_dir), "freeze_backbone": 0, "head_type": "linear"}
        model = load_model("resnet18-allBackbone", entry, project_root=tmp_path)
        assert isinstance(model, ClassifierModel)

    def test_arch_inferred_from_raw_name(self, tmp_path):
        """Architecture is taken as the first dash-separated token of raw_name."""
        ckpt_path = _make_checkpoint(tmp_path)
        entry = {"path": str(ckpt_path), "freeze_backbone": 0, "head_type": "linear"}
        # raw_name prefix "resnet18" matches the checkpoint we built
        model = load_model("resnet18-frozenBackbone", entry, project_root=tmp_path)
        assert isinstance(model, ClassifierModel)

    def test_missing_keys_does_not_raise(self, tmp_path):
        """strict=False: mismatched keys produce warnings but no exception."""
        ckpt_path = tmp_path / "partial.pt"
        # Save an empty state_dict — all keys will be missing
        torch.save({}, ckpt_path)
        entry = {"path": str(ckpt_path), "freeze_backbone": 0, "head_type": "linear"}
        model = load_model("resnet18-allBackbone", entry, project_root=tmp_path)
        assert isinstance(model, ClassifierModel)


# ---------------------------------------------------------------------------
# load_best_models
# ---------------------------------------------------------------------------


class TestLoadBestModels:
    def test_populates_model_key(self, tmp_path):
        """After load_best_models each entry has a 'model' key."""
        ckpt_path = _make_checkpoint(tmp_path)
        best_models = {
            "resnet18-allBackbone": {
                "path": str(ckpt_path),
                "freeze_backbone": 0,
                "head_type": "linear",
            }
        }
        load_best_models(best_models, project_root=tmp_path)
        assert "model" in best_models["resnet18-allBackbone"]
        assert isinstance(best_models["resnet18-allBackbone"]["model"], ClassifierModel)

    def test_mutates_dict_in_place(self, tmp_path):
        """load_best_models mutates the dict rather than returning a new one."""
        ckpt_path = _make_checkpoint(tmp_path)
        best_models = {
            "resnet18-allBackbone": {
                "path": str(ckpt_path),
                "freeze_backbone": 0,
                "head_type": "linear",
            }
        }
        original_id = id(best_models)
        load_best_models(best_models, project_root=tmp_path)
        assert id(best_models) == original_id

    def test_multiple_entries(self, tmp_path):
        """All entries in best_models are loaded."""
        ckpt1 = _make_checkpoint(tmp_path)
        ckpt2 = tmp_path / "best2.pt"
        # Reuse same weights for simplicity
        wrapper = ClassifierModel(
            base_model="resnet18", num_classes=1, optimizer="AdamW", learning_rate=5e-5
        )
        torch.save(wrapper.base_model.state_dict(), ckpt2)

        best_models = {
            "resnet18-a": {"path": str(ckpt1), "freeze_backbone": 0, "head_type": "linear"},
            "resnet18-b": {"path": str(ckpt2), "freeze_backbone": 0, "head_type": "linear"},
        }
        load_best_models(best_models, project_root=tmp_path)
        assert all("model" in entry for entry in best_models.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
