"""Configuration module for Ulcer Detection project."""

from .loader import Config, legacy_dict_to_config, load_config, load_model_config
from .mlflow_config import MLFlowConfig
from .models import MODEL_REGISTRY, ModelConfig, get_img_size, get_model_entry
from .paths import InformativePaths, MesPaths, PathConfig, UlcerPaths
from .training import CVConfig, EvaluationConfig, TrainingConfig
from .validation import validate_config

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "CVConfig",
    "EvaluationConfig",
    "PathConfig",
    "MesPaths",
    "UlcerPaths",
    "InformativePaths",
    "MLFlowConfig",
    "MODEL_REGISTRY",
    "get_img_size",
    "get_model_entry",
    "load_config",
    "load_model_config",
    "Config",
    "legacy_dict_to_config",
    "validate_config",
]
