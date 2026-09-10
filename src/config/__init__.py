"""Configuration module for Ulcer Detection project."""

from .loader import Config, load_config
from .mlflow_config import MLFlowConfig
from .models import MODEL_REGISTRY, ModelConfig, get_img_size, get_model_entry
from .paths import PathConfig, UlcerPaths
from .training import CVConfig, EvaluationConfig, TrainingConfig
from .validation import validate_config

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "CVConfig",
    "EvaluationConfig",
    "PathConfig",
    "UlcerPaths",
    "MLFlowConfig",
    "MODEL_REGISTRY",
    "get_img_size",
    "get_model_entry",
    "load_config",
    "Config",
    "validate_config",
]
