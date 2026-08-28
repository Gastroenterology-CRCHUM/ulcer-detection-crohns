"""MLflow configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLFlowConfig:
    """MLflow experiment and logging configuration."""

    experiment_name: str = "ulcer_detection"
    """Experiment name for main ulcer detection task."""

    log_model: bool = True
    """Log trained model artifacts to MLflow."""

    log_plots: bool = True
    """Log evaluation plots to MLflow."""

    log_confusion_matrix: bool = True
    """Log confusion matrix visualizations."""

    log_attention_maps: bool = False
    """Log attention map visualizations (can be memory-intensive)."""

    run_name_prefix: str = ""
    """Optional prefix for run names."""

    def __post_init__(self):
        """Validate configuration."""
        if not self.experiment_name:
            raise ValueError("experiment_name cannot be empty")
