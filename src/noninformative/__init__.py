"""src/noninformative — Non-Informative frame detection module."""

from src.noninformative.features import (
    FEATURE_NAMES,
    BottleneckExtractor,
    extract_all,
    extract_handcrafted,
    extract_handcrafted_batch,
)
from src.noninformative.model import NonInformativeClassifier
from src.noninformative.predict import (
    predict_dataframe,
    predict_video,
    sample_level_aggregation,
)

__all__ = [
    "extract_handcrafted",
    "extract_handcrafted_batch",
    "extract_all",
    "BottleneckExtractor",
    "FEATURE_NAMES",
    "NonInformativeClassifier",
    "predict_dataframe",
    "predict_video",
    "sample_level_aggregation",
]
