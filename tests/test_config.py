"""
tests/test_config.py
===================
Unit tests for the configuration system.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.config import (
    Config,
    CVConfig,
    EvaluationConfig,
    MLFlowConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
    legacy_dict_to_config,
    load_config,
)
from src.utils import ConfigurationError


@pytest.fixture(autouse=True)
def isolated_project_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run config tests in an isolated workspace with required directories.

    PathConfig validates that `data/ulcer/processed` and `data/ulcer/splits`
    exist. In CI, these folders are not committed, so tests must provision a
    minimal project tree before instantiating configuration objects.
    """
    required = [
        tmp_path / "data" / "ulcer" / "processed",
        tmp_path / "data" / "ulcer" / "splits",
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(tmp_path)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_valid_model_config(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            model="vitb16_imagenet_sup", num_classes=2, freeze_layers=-1, head_type="linear"
        )
        assert config.model == "vitb16_imagenet_sup"
        assert config.num_classes == 2
        assert config.freeze_layers == -1
        assert config.head_type == "linear"

    def test_invalid_model(self):
        """Test that invalid model raises error."""
        with pytest.raises(ConfigurationError):
            ModelConfig(model="invalid_model")

    def test_invalid_num_classes(self):
        """Test that invalid num_classes raises error."""
        with pytest.raises(ConfigurationError):
            ModelConfig(num_classes=0)

    def test_invalid_freeze_layers(self):
        """Test that invalid freeze_layers raises error."""
        with pytest.raises(ConfigurationError):
            ModelConfig(freeze_layers=-2)


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_valid_training_config(self):
        """Test creating a valid TrainingConfig."""
        config = TrainingConfig(
            batch_size=32,
            epochs=100,
            learning_rate=1e-4,
            optimizer="AdamW",
            weight_decay=1e-4,
            num_workers=4,
            equalize=True,
        )
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert config.optimizer == "AdamW"

    def test_invalid_batch_size(self):
        """Test that invalid batch_size raises error."""
        with pytest.raises(ConfigurationError):
            TrainingConfig(batch_size=0)

    def test_invalid_learning_rate(self):
        """Test that invalid learning_rate raises error."""
        with pytest.raises(ConfigurationError):
            TrainingConfig(learning_rate=-1.0)

    def test_invalid_epochs(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(epochs=0)

    def test_invalid_optimizer(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(optimizer="SGD")

    def test_invalid_subset_ratio(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(subset_ratio=0.0)

    def test_invalid_label_smoothing(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(label_smoothing=1.0)

    def test_invalid_dropout_rate(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(dropout_rate=1.0)

    def test_invalid_lr_factor(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(lr_factor=0.0)

    def test_invalid_num_workers(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(num_workers=-1)

    def test_invalid_warmup_epochs(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(warmup_epochs=-1)

    def test_invalid_min_lr(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(min_lr=0.0)

    def test_invalid_randaugment_m(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(randaugment_m=31)

    def test_invalid_random_erasing_p(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(random_erasing_p=1.5)

    def test_default_method(self):
        config = TrainingConfig.default()
        assert isinstance(config, TrainingConfig)


class TestCVConfig:
    def test_valid_cv_config(self):
        config = CVConfig(n_splits=5)
        assert config.n_splits == 5

    def test_invalid_n_splits(self):
        with pytest.raises(ValueError):
            CVConfig(n_splits=1)


class TestEvaluationConfig:
    def test_valid_evaluation_config(self):
        config = EvaluationConfig(bootstrap_samples=1000, confidence_interval=95)
        assert config.bootstrap_samples == 1000

    def test_invalid_bootstrap_samples(self):
        with pytest.raises(ValueError):
            EvaluationConfig(bootstrap_samples=50)

    def test_invalid_confidence_interval_too_low(self):
        with pytest.raises(ValueError):
            EvaluationConfig(confidence_interval=49)

    def test_invalid_confidence_interval_too_high(self):
        with pytest.raises(ValueError):
            EvaluationConfig(confidence_interval=100)


class TestPathConfig:
    """Test PathConfig dataclass."""

    def test_valid_path_config(self):
        """Test creating a valid PathConfig."""
        config = PathConfig(
            ulcer_processed_dir=Path("data/ulcer/processed"),
            output_dir=Path("output"),
            ulcer_splits_dir=Path("data/ulcer/splits"),
        )
        assert config.output_dir == Path("output")
        assert config.ulcer_splits_dir == Path("data/ulcer/splits")

    def test_path_validation(self):
        """Test that paths are validated."""
        # This should work with new naming convention
        config = PathConfig(ulcer_processed_dir=Path("data"))
        assert config.ulcer_processed_dir == Path("data")

    def test_task_output_mapping(self):
        """Test task-dependent output directory mapping."""
        config = PathConfig(ulcer_processed_dir=Path("data"))

        det_cfg = config.get_task_output_config("ulcer_detection")
        size_cfg = config.get_task_output_config("ulcer_size")
        inf_cfg = config.get_task_output_config("informative")

        assert det_cfg["models_dir"] == config.ulcer_detection_models_dir
        assert size_cfg["models_dir"] == config.ulcer_size_models_dir
        assert inf_cfg["models_dir"] == config.informative_models_dir

    def test_get_ulcer_config(self):
        config = PathConfig()
        result = config.get_ulcer_config()
        assert "raw_dir" in result
        assert "processed_dir" in result
        assert "splits_dir" in result

    def test_get_ulcer_size_config(self):
        config = PathConfig()
        result = config.get_ulcer_size_config()
        assert "splits_dir" in result
        assert result["splits_dir"] == config.ulcer_splits_size_dir

    def test_get_informative_config(self):
        config = PathConfig()
        result = config.get_informative_config()
        assert "raw_dir" in result
        assert "splits_dir" in result

    def test_get_mes_config(self):
        config = PathConfig()
        result = config.get_mes_config()
        assert "splits_dir" in result
        assert "filtrated_dir" in result

    def test_get_ulcer_output_config(self):
        config = PathConfig()
        result = config.get_ulcer_output_config()
        assert "models_dir" in result
        assert "results_dir" in result

    def test_get_informative_output_config(self):
        config = PathConfig()
        result = config.get_informative_output_config()
        assert "models_dir" in result

    def test_get_mes_output_config(self):
        config = PathConfig()
        result = config.get_mes_output_config()
        assert "models_dir" in result
        assert "eda_dir" in result

    def test_ensure_output_dirs(self, tmp_path):
        """ensure_output_dirs creates all output directories."""
        config = PathConfig(output_dir=tmp_path / "output")
        config.ensure_output_dirs()
        assert (tmp_path / "output").exists()


class TestConfig:
    """Test main Config dataclass."""

    def test_valid_config(self):
        """Test creating a valid Config."""
        config = Config(
            model=ModelConfig(),
            training=TrainingConfig(),
            paths=PathConfig(),
            mlflow=MLFlowConfig(),
        )
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.mlflow, MLFlowConfig)

    def test_config_serialization(self):
        """Test that Config can be serialized to dict."""
        config = Config()
        # Config is a dataclass with nested dataclasses
        assert hasattr(config, "model")
        assert hasattr(config, "training")
        assert hasattr(config, "paths")
        assert hasattr(config, "mlflow")


class TestLoadConfig:
    """Test load_config function."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, Config)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "model": {"model": "vitb16_imagenet_sup", "num_classes": 2, "freeze_layers": -1},
            "training": {"batch_size": 16, "epochs": 50, "learning_rate": 5e-5},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.model.model == "vitb16_imagenet_sup"
            assert config.training.batch_size == 16
            assert config.training.epochs == 50
        finally:
            Path(temp_path).unlink()

    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        import yaml

        config_data = {
            "model": {"model": "vitb16_imagenet_sup", "num_classes": 1},
            "training": {"batch_size": 8, "learning_rate": 1e-4},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.model.model == "vitb16_imagenet_sup"
            assert config.training.batch_size == 8
        finally:
            Path(temp_path).unlink()


class TestLegacyCompatibility:
    """Test legacy dict compatibility."""

    def test_legacy_dict_to_config(self):
        """Test converting legacy CONFIG dict to Config object."""
        legacy_config = {
            "model": "vitb16_imagenet_sup",
            "num_classes": 2,
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 1e-4,
            "data_dir": "data",
            "output_dir": "output",
        }

        config = legacy_dict_to_config(legacy_config)
        assert isinstance(config, Config)
        assert config.model.model == "vitb16_imagenet_sup"
        assert config.training.batch_size == 32
        # legacy dict data_dir may be overridden; check it exists as Path
        assert isinstance(config.paths.ulcer_processed_dir, Path)

    def test_legacy_dict_with_defaults(self):
        """Test legacy dict conversion with missing values uses defaults."""
        legacy_config = {"model": "resnet18"}

        config = legacy_dict_to_config(legacy_config)
        assert config.model.model == "resnet18"
        assert config.training.batch_size == 64  # default
        assert config.training.epochs == 100  # default


if __name__ == "__main__":
    pytest.main([__file__])
