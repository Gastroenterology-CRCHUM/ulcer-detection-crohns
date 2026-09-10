"""Configuration loading and management."""

from __future__ import annotations

import json
from pathlib import Path

from .mlflow_config import MLFlowConfig
from .models import ModelConfig
from .paths import PathConfig, get_default_paths
from .training import CVConfig, EvaluationConfig, TrainingConfig


class Config:
    """Global configuration container."""

    def __init__(
        self,
        model: ModelConfig | None = None,
        training: TrainingConfig | None = None,
        cv: CVConfig | None = None,
        evaluation: EvaluationConfig | None = None,
        paths: PathConfig | None = None,
        mlflow: MLFlowConfig | None = None,
    ):
        """Initialize configuration."""
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.cv = cv or CVConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.paths = paths or get_default_paths()
        self.mlflow = mlflow or MLFlowConfig()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Config(\n"
            f"  model={self.model},\n"
            f"  training={self.training},\n"
            f"  cv={self.cv},\n"
            f"  evaluation={self.evaluation},\n"
            f"  paths={self.paths},\n"
            f"  mlflow={self.mlflow}\n"
            f")"
        )


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to config JSON/YAML file (optional).

    Returns:
        Loaded Config instance.
    """
    if config_path is None:
        cfg = Config()
        _validate_loaded_config(cfg)
        return cfg

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix == ".json":
        cfg = _load_json_config(config_path)
        _validate_loaded_config(cfg)
        return cfg
    elif config_path.suffix in (".yaml", ".yml"):
        cfg = _load_yaml_config(config_path)
        _validate_loaded_config(cfg)
        return cfg
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def _validate_loaded_config(config: Config) -> None:
    """Validate loaded configuration with lazy import to avoid import cycles."""
    from .validation import validate_config

    validate_config(config)


def _build_config_from_dict(data: dict) -> Config:
    return Config(
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        cv=CVConfig(**data.get("cv", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
        paths=PathConfig(**data.get("paths", {})),
        mlflow=MLFlowConfig(**data.get("mlflow", {})),
    )


def _load_json_config(path: Path) -> Config:
    """Load configuration from JSON file."""
    with open(path) as f:
        return _build_config_from_dict(json.load(f))


def _load_yaml_config(path: Path) -> Config:
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config files. Install with: pip install pyyaml"
        ) from None

    with open(path) as f:
        return _build_config_from_dict(yaml.safe_load(f) or {})


def config_to_dict(config: Config) -> dict:
    """Convert Config object back to old dict format for mlflow logging."""

    result = {
        "model": config.model.model,
        "num_classes": config.model.num_classes,
        "freeze_layers": config.model.freeze_layers,
        "threshold": config.model.threshold,
        "dropout_rate": config.model.dropout_rate,
        "head_type": config.model.head_type,
        "batch_size": config.training.batch_size,
        "epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "optimizer": config.training.optimizer,
        "weight_decay": config.training.weight_decay,
        "label_smoothing": config.training.label_smoothing,
        "class_weights": config.training.class_weights,
        "lr_patience": config.training.lr_patience,
        "lr_factor": config.training.lr_factor,
        "es_patience": config.training.es_patience,
        "equalize": config.training.equalize,
        "num_workers": config.training.num_workers,
        "device": config.training.device_id,
        "warmup_epochs": config.training.warmup_epochs,
        "min_lr": config.training.min_lr,
        "use_randaugment": config.training.use_randaugment,
        "randaugment_m": config.training.randaugment_m,
        "use_random_erasing": config.training.use_random_erasing,
        "random_erasing_p": config.training.random_erasing_p,
    }
    return result
