"""Pipeline B — Ulcer detection training / evaluation.

Usage
-----
    python scripts/ulcer/train.py --mode split
    python scripts/ulcer/train.py --mode cv --use-full-trainset
    python scripts/ulcer/train.py --mode split --register
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from src.config import load_config
from src.training.run_modes import PipelineDef, run_cv_mode, run_split_mode, setup_training
from src.utils import setup_logging

setup_logging("ulcer_detection", log_dir=Path("logs"))
cfg = load_config()

PIPELINE = PipelineDef(
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

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VALIDATING CONFIGURATION — Pipeline B (Ulcer Detection)")
    print("=" * 80)

    args, device, img_size, num_workers = setup_training(
        cfg, description="Train ulcer detection model"
    )

    data_dir = cfg.paths.ulcer_processed_dir
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else cfg.paths.ulcer_splits_dir / "dataset_manifest.csv"
    )

    mlflow.end_run()
    mlflow.set_tracking_uri(cfg.paths.mlflow_db)
    mlflow.set_experiment(PIPELINE.experiment_name)

    if args.mode == "split":
        run_split_mode(
            cfg, PIPELINE, device, manifest_path, data_dir, num_workers, img_size, args.register
        )
    else:
        run_cv_mode(
            cfg,
            PIPELINE,
            device,
            manifest_path,
            data_dir,
            num_workers,
            img_size,
            args.n_splits,
            args.use_full_trainset,
            args.fold,
            args.register,
        )
