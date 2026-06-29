"""Utilities package - Common helpers and utilities."""

from .common import (
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError,
    UlcerDetectionError,
    compute_confidence_interval,
    ensure_dir,
    find_latest_checkpoint,
    format_metrics,
    get_device,
    get_device_info,
    loader_dataset_size,
    set_seed,
    validate_file_exists,
    validate_path_exists,
    validate_value_range,
)
from .logging import get_logger, setup_logging

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # Exceptions
    "UlcerDetectionError",
    "ConfigurationError",
    "DataError",
    "ModelError",
    "TrainingError",
    # Device utilities
    "get_device",
    "get_device_info",
    "set_seed",
    # Path utilities
    "ensure_dir",
    "find_latest_checkpoint",
    # Metrics utilities
    "compute_confidence_interval",
    "format_metrics",
    # DataLoader helpers
    "loader_dataset_size",
    # Validators
    "validate_path_exists",
    "validate_file_exists",
    "validate_value_range",
]
