"""Common utilities and helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Custom Exceptions
# ═══════════════════════════════════════════════════════════════════════════════


class UlcerDetectionError(Exception):
    """Base exception for ulcer detection errors."""

    pass


class ConfigurationError(UlcerDetectionError):
    """Raised when configuration is invalid."""

    pass


class DataError(UlcerDetectionError):
    """Raised when data loading/processing fails."""

    pass


class ModelError(UlcerDetectionError):
    """Raised when model-related errors occur."""

    pass


class TrainingError(UlcerDetectionError):
    """Raised when training fails."""

    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Device & Hardware Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def get_device(device_id: int = 0) -> torch.device:
    """Get safe device handle.

    Args:
        device_id: GPU ID (< 0 for CPU).

    Returns:
        torch.device instance.
    """
    if device_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")

    if device_id >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {device_id} not available. Available GPUs: {torch.cuda.device_count()}"
        )

    return torch.device(f"cuda:{device_id}")


def get_device_info() -> dict[str, Any]:
    """Get device information."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Path Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def ensure_dir(path: Path | str) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_checkpoint(model_dir: Path):
    """Find most recent checkpoint in model directory.

    Args:
        model_dir: Model directory path.

    Returns:
        Path to latest checkpoint, or None if not found.
    """
    if not model_dir.exists():
        return None

    checkpoints = list(model_dir.glob("*/*/best.pt"))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def compute_confidence_interval(
    metric_values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute confidence interval for a metric.

    Args:
        metric_values: Array of metric values.
        confidence: Confidence level (0 to 1).

    Returns:
        (mean, lower_bound, upper_bound)
    """
    mean = np.mean(metric_values)
    std_err = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))
    margin = 1.96 * std_err * np.sqrt(-2 * np.log(1 - confidence))
    return mean, mean - margin, mean + margin


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    """Format metrics for display.

    Args:
        metrics: Dictionary of metrics.
        prefix: Optional prefix for each metric name.

    Returns:
        Formatted string.
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {prefix}{key}: {value:.4f}")
        else:
            lines.append(f"  {prefix}{key}: {value}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader helpers
# ═══════════════════════════════════════════════════════════════════════════════


def loader_dataset_size(loader) -> int:
    """Return the number of samples in a DataLoader's underlying dataset."""
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "__len__"):
        return int(len(dataset))
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Type Validators
# ═══════════════════════════════════════════════════════════════════════════════


def validate_path_exists(path: Path | str, name: str = "path") -> Path:
    """Validate that a path exists.

    Args:
        path: Path to validate.
        name: Name for error messages.

    Returns:
        Path object.

    Raises:
        DataError: If path doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"{name} does not exist: {path}")
    return path


def validate_file_exists(path: Path | str, name: str = "file") -> Path:
    """Validate that a file exists.

    Args:
        path: File path to validate.
        name: Name for error messages.

    Returns:
        Path object.

    Raises:
        DataError: If file doesn't exist.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{name} does not exist: {path}")
    return path


def validate_value_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value",
) -> float:
    """Validate that a value is within a range.

    Args:
        value: Value to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        name: Name for error messages.

    Returns:
        The value if valid.

    Raises:
        ConfigurationError: If value is out of range.
    """
    if not min_val <= value <= max_val:
        raise ConfigurationError(f"{name} must be in [{min_val}, {max_val}], got {value}")
    return value
