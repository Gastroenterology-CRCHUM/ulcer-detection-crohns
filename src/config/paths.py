"""Path configuration and management.

This module provides centralized path management for all data, output, and asset directories.

Data Directory Structure
========================

data/
├── ulcer/                         # Ulcer detection pipeline
│   ├── raw/                       # Source frames (1920×1080 JPEG)
│   │   ├── Ulcer/
│   │   │   └── vid_XX_XXXX/
│   │   │       └── ulcer_X/
│   │   ├── NonUlcer/
│   │   │   └── vid_XX_XXXX/
│   │   │       └── normal_X/
│   │   ├── videos/                # Original videos (annotation extraction)
│   │   └── Ulcer and Non-Ulcer Timestamps.xlsx
│   ├── processed/                 # Cropped frames (1350×1080)
│   │   ├── Ulcer/
│   │   └── NonUlcer/
│   └── splits/                    # Train/val/test manifests
│       ├── dataset_manifest.csv
│       ├── split_info.json
│       └── train.csv / val.csv / test.csv
│
├── informative/                   # Informative/Non-Informative frame classification pipeline
│   ├── raw/                       # Source frames (1920×1080 JPEG)
│   │   ├── Informative/
│   │   └── Non-Informative/
│   │       ├── Blur/
│   │       ├── Low light/
│   │       ├── Debris/
│   │       └── ...
│   ├── processed/                 # Cropped frames (1350×1080)
│   └── splits/                    # Train/val/test manifests
│
└── assets/                        # Shared assets
    ├── pretrained/                # Pre-trained model weights (ResNet, ViT, DINO, etc.)
    └── informative/               # Trained informative-filter artifacts
        ├── rf_pipeline.pkl
        └── features_cache.pkl

output/
├── ulcer/
│   └── models/
│       └── detection/             # Checkpoints per model/timestamp
├── informative/
│   └── models/
└── shared/

results/
├── ulcer/
│   ├── cv/                        # CV result figures and tables
│   └── eda/                       # EDA figures and reports
└── informative/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils import ConfigurationError


# ============================================================================
# Per-pipeline path groups
# ============================================================================


@dataclass
class UlcerPaths:
    """All data directories for the Ulcer detection pipeline."""

    root: Path = field(default_factory=lambda: Path("data/ulcer"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def filtrated(self) -> Path:
        return self.root / "filtrated"

    @property
    def splits(self) -> Path:
        return self.root / "splits"


@dataclass
class InformativePaths:
    """All data directories for the Informative/Non-Informative pipeline."""

    root: Path = field(default_factory=lambda: Path("data/informative"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def splits(self) -> Path:
        return self.root / "splits"


# ============================================================================
# Central config
# ============================================================================


@dataclass
class PathConfig:
    """Central path configuration for all project directories.

    Data paths are grouped by pipeline via sub-objects::

        cfg.paths.ulcer.splits      # data/ulcer/splits
        cfg.paths.ulcer.filtrated   # data/ulcer/filtrated
        cfg.paths.informative.raw   # data/informative/raw
    """

    # ── Grouped data paths ────────────────────────────────────────────────────
    ulcer:       UlcerPaths       = field(default_factory=UlcerPaths)
    informative: InformativePaths = field(default_factory=InformativePaths)

    # ── Convenience aliases ───────────────────────────────────────────────────
    @property
    def ulcer_raw_dir(self) -> Path:           return self.ulcer.raw
    @property
    def ulcer_processed_dir(self) -> Path:     return self.ulcer.processed
    @property
    def ulcer_filtrated_dir(self) -> Path:     return self.ulcer.filtrated
    @property
    def ulcer_splits_dir(self) -> Path:        return self.ulcer.splits

    @property
    def informative_raw_dir(self) -> Path:       return self.informative.raw
    @property
    def informative_processed_dir(self) -> Path: return self.informative.processed
    @property
    def informative_splits_dir(self) -> Path:    return self.informative.splits

    # ============================================================================
    # ASSETS
    # ============================================================================
    pretrained_dir: Path = Path("data/assets/pretrained")
    gastronet_weights_dir: Path = Path("data/assets/pretrained")
    informative_assets_dir: Path = Path("data/assets/informative")
    informative_models_dir: Path = Path("data/assets/informative")
    informative_model_path: Path = Path("data/assets/informative/rf_pipeline.pkl")
    informative_features_cache: Path = Path("data/assets/informative/features_cache.pkl")

    # ============================================================================
    # OUTPUT DIRECTORIES
    # ============================================================================
    output_dir: Path = Path("output")
    output_ulcer_dir: Path = Path("output/ulcer")
    output_informative_dir: Path = Path("output/informative")
    output_shared_dir: Path = Path("output/shared")

    ulcer_models_root_dir: Path = Path("output/ulcer/models")
    ulcer_detection_models_dir: Path = Path("output/ulcer/models/detection")
    filtered_dir: Path = Path("output/ulcer/filtered")

    # ============================================================================
    # RESULTS DIRECTORIES
    # ============================================================================
    results_root_dir: Path = Path("results")
    results_ulcer_dir: Path = Path("results/ulcer")
    results_informative_dir: Path = Path("results/informative")
    results_eda_dir: Path = Path("results/ulcer/eda")
    results_cv_dir: Path = Path("results/ulcer/cv")

    # ============================================================================
    # MLflow
    # ============================================================================
    mlflow_dir: Path = Path("mlruns")
    mlflow_db: str = "sqlite:///mlflow.db"

    def __post_init__(self):
        essential_paths = [
            self.ulcer_splits_dir,
        ]
        for path in essential_paths:
            path = Path(path)
            if not path.exists():
                alt_path = Path("..") / path
                if not alt_path.exists():
                    raise ConfigurationError(f"Required path does not exist: {path.absolute()}")

    def ensure_output_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        output_paths = [
            self.output_dir,
            self.output_ulcer_dir,
            self.output_informative_dir,
            self.output_shared_dir,
            self.ulcer_models_root_dir,
            self.ulcer_detection_models_dir,
            self.informative_models_dir,
            self.filtered_dir,
            self.results_root_dir,
            self.results_ulcer_dir,
            self.results_informative_dir,
            self.results_eda_dir,
            self.results_cv_dir,
            self.ulcer_processed_dir,
        ]
        for path in output_paths:
            path.mkdir(parents=True, exist_ok=True)

    def get_ulcer_config(self) -> dict:
        """Get path configuration for the ulcer detection pipeline."""
        return {
            "raw_dir": self.ulcer_raw_dir,
            "processed_dir": self.ulcer_processed_dir,
            "splits_dir": self.ulcer_splits_dir,
        }

    def get_informative_config(self) -> dict:
        """Get path configuration for the informative frame classification pipeline."""
        return {
            "raw_dir": self.informative_raw_dir,
            "processed_dir": self.informative_processed_dir,
            "splits_dir": self.informative_splits_dir,
        }

    def get_ulcer_output_config(self) -> dict:
        """Get output/result directories for ulcer workflows."""
        return {
            "output_dir": self.output_ulcer_dir,
            "models_dir": self.ulcer_detection_models_dir,
            "filtered_dir": self.filtered_dir,
            "results_dir": self.results_ulcer_dir,
            "eda_dir": self.results_eda_dir,
            "cv_dir": self.results_cv_dir,
        }

    def get_informative_output_config(self) -> dict:
        """Get output/result directories for informative workflows."""
        return {
            "output_dir": self.output_informative_dir,
            "models_dir": self.informative_models_dir,
            "results_dir": self.results_informative_dir,
        }

    def get_task_output_config(self, task: str) -> dict:
        """Return output/results paths for the requested task.

        Supported tasks: ulcer_detection, informative, ulcer_filtering, eda
        """
        task_key = task.strip().lower()
        mapping = {
            "ulcer_detection": {
                "output_dir": self.output_ulcer_dir,
                "models_dir": self.ulcer_detection_models_dir,
                "results_dir": self.results_ulcer_dir,
            },
            "informative": {
                "output_dir": self.output_informative_dir,
                "models_dir": self.informative_models_dir,
                "results_dir": self.results_informative_dir,
            },
            "ulcer_filtering": {
                "output_dir": self.filtered_dir,
                "models_dir": self.informative_models_dir,
                "results_dir": self.results_ulcer_dir,
            },
            "eda": {
                "output_dir": self.results_eda_dir,
                "results_dir": self.results_eda_dir,
            },
        }

        if task_key not in mapping:
            allowed = ", ".join(sorted(mapping.keys()))
            raise ValueError(f"Unknown task '{task}'. Supported tasks: {allowed}")

        return mapping[task_key]


def get_default_paths() -> PathConfig:
    """Get default path configuration."""
    return PathConfig()
