"""Ulcer detection training / evaluation.

Reproducing paper results
-------------------------
    python -m scripts.ulcer.train --mode cv

Fine-tuning GastroNet-5M for a new task
---------------------------------------
    python -m scripts.ulcer.train \
        --config configs/your_task.yaml \
        --manifest data/your_task/splits/manifest.csv \
        --data-dir data/your_task/processed \
        --mode cv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow

from src.config import load_config
from src.training.run_modes import PipelineDef, run_cv_mode, run_split_mode, setup_training
from src.utils import setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ulcer detection model (or fine-tune for a new task)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce paper results (5-fold CV)
  python -m scripts.ulcer.train --mode cv

  # Fine-tune GastroNet-5M for a new task
  python -m scripts.ulcer.train \\
      --config configs/your_task.yaml \\
      --manifest data/your_task/splits/manifest.csv \\
      --data-dir data/your_task/processed \\
      --mode cv
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: use built-in defaults for paper reproduction)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory (default: data/ulcer/processed)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override manifest path (default: data/ulcer/splits/dataset_manifest.csv)",
    )
    parser.add_argument("--mode", choices=["cv", "split"], default="split")
    parser.add_argument("--fold", type=int, default=None, help="Run only this fold (CV mode)")
    parser.add_argument("--n-splits", type=int, default=None, help="Number of CV folds (default: 5)")
    parser.add_argument("--use-full-trainset", action="store_true", help="Merge val into train for CV")
    parser.add_argument("--register", action="store_true", help="Register model in MLflow Registry")
    return parser.parse_args()


def main():
    setup_logging("ulcer_detection", log_dir=Path("logs"))

    # Parse arguments first (so --config can be used)
    args = parse_args()

    # Load config (from file if specified, otherwise defaults)
    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path)

    # Override n_splits if specified
    if args.n_splits is not None:
        cfg.cv.n_splits = args.n_splits

    # Determine data directory and manifest path
    data_dir = Path(args.data_dir) if args.data_dir else cfg.paths.ulcer_processed_dir
    manifest_path = (
        Path(args.manifest) if args.manifest else cfg.paths.ulcer_splits_dir / "dataset_manifest.csv"
    )

    # Validate paths exist
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)

    # Build pipeline definition
    pipeline = PipelineDef(
        label_col="label",
        num_classes=cfg.model.num_classes,
        models_root=cfg.paths.get_task_output_config("ulcer_detection")["models_dir"],
        experiment_name=cfg.mlflow.experiment_name,
        registry_prefix="ulcer_",
        run_name_infix="",
        aggregate_by_clip=cfg.training.aggregate_by_clip,
        tune_threshold=True,
        is_multiclass=False,
        pipeline_tag="B_ulcer",
        comparison_metrics=["test__f1_mean", "test__auroc_mean", "test_clip_f1"],
    )

    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"  Config:   {args.config or '(defaults)'}")
    print(f"  Model:    {cfg.model.model}")
    print(f"  Data dir: {data_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Mode:     {args.mode}")
    if args.mode == "cv":
        print(f"  Folds:    {cfg.cv.n_splits}")
    print("=" * 80 + "\n")

    # Setup device and training params
    from src.config import get_img_size
    from src.utils import get_device
    import torch
    import os

    device = get_device()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(cfg.training.device_id)}")
    torch.backends.cudnn.benchmark = True

    img_size = get_img_size(cfg.model.model)
    num_workers = min(cfg.training.num_workers, os.cpu_count() or 8)

    # MLflow setup
    mlflow.end_run()
    mlflow.set_tracking_uri(cfg.paths.mlflow_db)
    mlflow.set_experiment(pipeline.experiment_name)

    # Run training
    if args.mode == "split":
        run_split_mode(
            cfg, pipeline, device, manifest_path, data_dir, num_workers, img_size, args.register
        )
    else:
        run_cv_mode(
            cfg,
            pipeline,
            device,
            manifest_path,
            data_dir,
            num_workers,
            img_size,
            cfg.cv.n_splits,
            args.use_full_trainset,
            args.fold,
            args.register,
        )


if __name__ == "__main__":
    main()
