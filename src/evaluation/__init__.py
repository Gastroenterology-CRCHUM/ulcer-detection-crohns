"""
src/evaluation/__init__.py
--------------------------
Public API for the evaluation module.

Usage
-----
    from src.evaluation import compute_metrics_with_ci, plot_roc_curves
    from src.evaluation import log_evaluation
"""

from src.evaluation.aggregation import aggregate_frame_to_clip
from src.evaluation.metrics import (
    METRIC_FNS,
    bootstrap_ci,
    bootstrap_ci_aggregated,
    compute_clip_metrics,
    compute_metrics_with_ci,
    fmt,
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_delong_heatmap,
    plot_roc_curve,
    plot_roc_curves,
)
__all__ = [
    # metrics
    "METRIC_FNS",
    "bootstrap_ci",
    "bootstrap_ci_aggregated",
    "compute_clip_metrics",
    "compute_metrics_with_ci",
    "fmt",
    "aggregate_frame_to_clip",
    # plots
    "plot_confusion_matrix",
    "plot_delong_heatmap",
    "plot_roc_curve",
    "plot_roc_curves",
]
