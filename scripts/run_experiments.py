"""
scripts/run_experiments.py
==========================
Experiment orchestrator — runs sequential training jobs from a YAML plan.

Usage
-----
    # Ulcer detection
    python scripts/run_experiments.py --plan configs/experiments/ulcer_batch.yaml

    # Dry-run (print plan, no training)
    python scripts/run_experiments.py --dry-run

    # Single model from a plan
    python scripts/run_experiments.py --plan ... --model vits16_gastronet

YAML Plan Format
----------------
    runs:
      - model: vits16_gastronet
        freeze_layers: 0        # 0=full finetune | -1=freeze backbone | N=first-N blocks
        lr: 1.0e-4
        batch_size: 64
        epochs: 100
        mode: cv                # split | cv  (default: split)
        dropout_rate: 0.3
        weight_decay: 1.0e-2
        label_smoothing: 0.0
        n_splits: 5             # CV folds when mode=cv
        register: false         # register in MLflow Model Registry
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import time
import traceback
import warnings
from dataclasses import replace as dc_replace
from datetime import timedelta
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, Config, load_config  # noqa: E402
from src.training.run_modes import PipelineDef, run_cv_mode, run_split_mode  # noqa: E402

warnings.filterwarnings("ignore", message=".*local version label.*")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Run defaults
# ---------------------------------------------------------------------------

RUN_DEFAULTS: dict = {
    "mode": "cv",
    "freeze_layers": 0,
    "lr": 1e-4,
    "batch_size": 64,
    "epochs": 100,
    "dropout_rate": 0.5,
    "weight_decay": 0.01,
    "label_smoothing": 0.0,
    "n_splits": 5,
    "register": False,
    "tune_threshold": True,
    "tune_clip_threshold": False,
}

# Built-in plan — 9 model configurations evaluated in the paper
# (Berndt*, Mashayekhi* et al. — ulcer detection in Crohn's disease)
# All runs use 5-fold patient-stratified cross-validation.
# LRs: self-supervised models confirmed from sibling data-efficiency repo.
# Supervised ImageNet LRs follow standard fine-tuning convention (1e-4 / 1e-5).
DEFAULT_PLAN: list[dict] = [
    # ── ResNet-50 ────────────────────────────────────────────────────────
    {"model": "resnet50_imagenet_sup", "freeze_layers": 0, "lr": 1e-4,  "epochs": 100, "batch_size": 64},
    {"model": "resnet50_imagenet",     "freeze_layers": 0, "lr": 1e-5,  "epochs": 100, "batch_size": 64},
    {"model": "resnet50_gastronet",    "freeze_layers": 0, "lr": 1e-6,  "epochs": 100, "batch_size": 64},
    # ── EfficientNet-B0 ──────────────────────────────────────────────────
    {"model": "efficientnetb0",        "freeze_layers": 0, "lr": 3e-5,  "epochs": 100, "batch_size": 64},
    # ── ViT-Base/16 ──────────────────────────────────────────────────────
    {"model": "vitb16_imagenet_sup",   "freeze_layers": 0, "lr": 1e-4,  "epochs": 100, "batch_size": 64},
    {"model": "vitb16_imagenet",       "freeze_layers": 0, "lr": 1e-6,  "epochs": 100, "batch_size": 64},
    # ── ViT-Small/16 ─────────────────────────────────────────────────────
    {"model": "vits16_imagenet_hf",    "freeze_layers": 0, "lr": 1e-5,  "epochs": 100, "batch_size": 64},
    {"model": "vits16_imagenet",       "freeze_layers": 0, "lr": 1e-6,  "epochs": 100, "batch_size": 64},
    {"model": "vits16_gastronet",      "freeze_layers": 0, "lr": 1e-6,  "epochs": 100, "batch_size": 64},
]


# ---------------------------------------------------------------------------
# Per-task configuration
# ---------------------------------------------------------------------------

_TASK_DISPLAY = {
    "ulcer": "Ulcer Detection",
}


def _get_task_paths(task: str, cfg: Config, manifest_override: str | None) -> tuple[Path, Path]:
    """Return (manifest_path, data_dir) for the ulcer task."""
    manifest = cfg.paths.ulcer_splits_dir / "dataset_manifest.csv"
    data_dir = cfg.paths.ulcer_processed_dir
    if manifest_override:
        manifest = Path(manifest_override)
    return manifest, data_dir


def _get_experiment_name(task: str, cfg: Config) -> str:
    return cfg.mlflow.experiment_name


def _get_models_root(task: str, cfg: Config) -> Path:
    return cfg.paths.get_task_output_config("ulcer_detection")["models_dir"]


def _build_pipeline(
    task: str, cfg: Config, manifest_df: pd.DataFrame, num_classes: int | None = None
) -> PipelineDef:
    """Build a PipelineDef for ulcer detection."""
    label_col = "label"
    num_classes = num_classes if num_classes is not None else 1

    return PipelineDef(
        label_col=label_col,
        num_classes=num_classes,
        models_root=_get_models_root(task, cfg),
        experiment_name=_get_experiment_name(task, cfg),
        registry_prefix="ulcer_",
        run_name_infix="",
        aggregate_by_clip=cfg.training.aggregate_by_clip,
        tune_threshold=True,
        tune_clip_threshold=False,
        is_multiclass=False,
        pipeline_tag="ulcer",
        class_names=None,
        use_size_distribution=False,
        comparison_file_suffix="",
        comparison_metrics=["test__f1_mean", "test__auroc_mean", "test_clip_f1"],
        extra_params={"pipeline": "ulcer"},
    )


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------


def _apply_run_config(base_config: Config, run: dict) -> Config:
    new_training = dc_replace(
        base_config.training,
        epochs=run["epochs"],
        learning_rate=run["lr"],
        batch_size=run["batch_size"],
        dropout_rate=run.get("dropout_rate", base_config.training.dropout_rate),
        weight_decay=run.get("weight_decay", base_config.training.weight_decay),
        label_smoothing=run.get("label_smoothing", base_config.training.label_smoothing),
        num_workers=run.get("num_workers", base_config.training.num_workers),
        warmup_epochs=run.get("warmup_epochs", base_config.training.warmup_epochs),
        min_lr=run.get("min_lr", base_config.training.min_lr),
        use_randaugment=run.get("use_randaugment", base_config.training.use_randaugment),
        randaugment_m=run.get("randaugment_m", base_config.training.randaugment_m),
        use_random_erasing=run.get("use_random_erasing", base_config.training.use_random_erasing),
        random_erasing_p=run.get("random_erasing_p", base_config.training.random_erasing_p),
    )
    new_model = dc_replace(
        base_config.model,
        model=run["model"],
        freeze_layers=run["freeze_layers"],
    )
    return Config(
        model=new_model,
        training=new_training,
        cv=base_config.cv,
        evaluation=base_config.evaluation,
        paths=base_config.paths,
        mlflow=base_config.mlflow,
    )


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------


def load_plan(yaml_path: Path) -> list[dict]:
    raw = yaml.safe_load(yaml_path.read_text())
    runs = raw.get("runs", [])
    if not runs:
        raise ValueError(f"No runs defined in {yaml_path}")
    base = {**RUN_DEFAULTS, **raw.get("defaults", {})}
    return [{**base, **r} for r in runs]


def build_plan(model_filter: str | None = None) -> list[dict]:
    plan = [{**RUN_DEFAULTS, **r} for r in DEFAULT_PLAN]
    if model_filter:
        plan = [r for r in plan if r["model"] == model_filter]
    return plan


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def _fmt_run(i: int, total: int, run: dict) -> str:
    freeze_label = {0: "allBackbone", -1: "freezeBackbone"}.get(
        run["freeze_layers"], f"{run['freeze_layers']}Backbone"
    )
    return (
        f"[{i}/{total}]  {run['model']:<25}  {freeze_label:<16}  "
        f"lr={run['lr']:.0e}  bs={run['batch_size']}  ep={run['epochs']}"
    )


def print_plan(task: str, plan: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"EXPERIMENT PLAN  —  {_TASK_DISPLAY[task]}  —  {len(plan)} runs")
    print("=" * 80)
    for i, run in enumerate(plan, 1):
        print(_fmt_run(i, len(plan), run))
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Metrics report
# ---------------------------------------------------------------------------


def _print_metrics_report(pipeline: PipelineDef, session_start: float) -> None:
    """Query MLflow for parent runs started in this session and print a metrics table."""
    try:
        experiment = mlflow.get_experiment_by_name(pipeline.experiment_name)
        if experiment is None:
            return

        start_ms = int(session_start * 1000)
        runs_df = pd.DataFrame(
            mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"start_time >= {start_ms}",
                order_by=["start_time ASC"],
                output_format="pandas",
            )
        )
        if runs_df.empty:
            return

        parent_col = "tags.mlflow.parentRunId"
        if parent_col in runs_df.columns:
            runs_df = runs_df[runs_df[parent_col].isna() | (runs_df[parent_col] == "")]
        if runs_df.empty:
            return

        client = mlflow.MlflowClient()

        def _load_ci(run_id: str, artifact: str) -> dict:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    path = client.download_artifacts(run_id, artifact)
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return {}

        def _m(row: pd.Series, key: str, w: int = 7) -> str:
            val = row.get(f"metrics.{key}")
            return f"{float(val):{w}.4f}" if pd.notna(val) else " " * (w - 1) + "—"

        def _ci_fmt(ci: dict, name: str, w: int = 8) -> str:
            entry = ci.get(name, {})
            lo, hi = entry.get("lower"), entry.get("upper")
            if lo is not None and hi is not None:
                half = (hi - lo) / 2
                return f"(±{half:.3f})"
            return ""

        def _is_cv(row: pd.Series) -> bool:
            return pd.notna(row.get("metrics.cv_mean_val_auroc"))

        def _thr(row: pd.Series) -> str:
            keys = (
                ("cv_mean_threshold", "cv_final_threshold")
                if _is_cv(row)
                else ("test_threshold", "best_threshold")
            )
            for key in keys:
                val = row.get(f"metrics.{key}")
                if pd.notna(val):
                    return f"{float(val):6.4f}"
            return "  — "

        def _display_name(row: pd.Series) -> str:
            name = str(row.get("tags.mlflow.runName", "?"))
            for suffix in ("_split", "_cv"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            return name[:31]

        W = 87

        # ── Frame-level ────────────────────────────────────────────────────────
        print("\n" + "=" * W)
        print("PERFORMANCE REPORT — FRAME LEVEL")
        print("=" * W)
        hdr = (
            f"{'Model':<32}"
            f" {'Thr':^6} {'F1':^7} {'F1@0.5':^7} {'Sens':^7} {'Spec':^7}"
            f" {'Acc':^7} {'AUROC':^7}"
        )
        print(hdr)
        print("-" * W)
        for _, row in runs_df.iterrows():
            run_id = row["run_id"]
            if _is_cv(row):
                ci = _load_ci(run_id, "metrics/best_fold_ci.json")
                print(
                    f"{_display_name(row):<32}"
                    f" {_thr(row):^6}"
                    f" {_m(row, 'cv_mean_val_f1'):^7}"
                    f" {'—':^7}"
                    f" {_m(row, 'cv_mean_val_recall'):^7}"
                    f" {'—':^7}"
                    f" {'—':^7}"
                    f" {_m(row, 'cv_mean_val_auroc'):^7}"
                )
                ci_line = (
                    f"{'':39}"
                    f" {_ci_fmt(ci, 'F1'):^8}"
                    f" {'':^7}"
                    f" {_ci_fmt(ci, 'Sensitivity'):^8}"
                    f" {'':^8}"
                    f" {'':^8}"
                    f" {_ci_fmt(ci, 'AUROC'):^8}"
                )
            else:
                ci = _load_ci(run_id, "metrics/test_ci.json")
                print(
                    f"{_display_name(row):<32}"
                    f" {_thr(row):^6}"
                    f" {_m(row, 'test__f1_mean'):^7}"
                    f" {_m(row, 'test_f1_at05'):^7}"
                    f" {_m(row, 'test__sensitivity_mean'):^7}"
                    f" {_m(row, 'test__specificity_mean'):^7}"
                    f" {_m(row, 'test__accuracy_mean'):^7}"
                    f" {_m(row, 'test__auroc_mean'):^7}"
                )
                ci_line = (
                    f"{'':39}"
                    f" {_ci_fmt(ci, 'F1'):^8}"
                    f" {'':^7}"
                    f" {_ci_fmt(ci, 'Sensitivity'):^8}"
                    f" {_ci_fmt(ci, 'Specificity'):^8}"
                    f" {_ci_fmt(ci, 'Accuracy'):^8}"
                    f" {_ci_fmt(ci, 'AUROC'):^8}"
                )
            if ci_line.strip():
                print(ci_line)
        print("=" * W)

        # ── Clip-level ─────────────────────────────────────────────────────────
        has_clip = any(
            pd.notna(row.get("metrics.test_clip__f1_mean")) for _, row in runs_df.iterrows()
        )
        if has_clip:
            print("\n" + "=" * W)
            print("PERFORMANCE REPORT — CLIP LEVEL")
            print("=" * W)
            hdr_clip = (
                f"{'Model':<32} {'Clip F1':^9} {'Clip AUROC':^11}"
                f" {'Clip Sens':^10} {'Clip Prec':^10} {'N clips':^8}"
            )
            print(hdr_clip)
            print("-" * W)
            for _, row in runs_df.iterrows():
                run_id = row["run_id"]
                ci_clip = _load_ci(run_id, "metrics/test_clip_ci.json")
                print(
                    f"{_display_name(row):<32}"
                    f" {_m(row, 'test_clip__f1_mean'):^9}"
                    f" {_m(row, 'test_clip__auroc_mean'):^11}"
                    f" {_m(row, 'test_clip__sensitivity_mean'):^10}"
                    f" {_m(row, 'test_clip__precision_mean'):^10}"
                    f" {_m(row, 'test_clip_n_clips', 3):^8}"
                )
                ci_line_clip = (
                    f"{'':32}"
                    f" {_ci_fmt(ci_clip, 'F1'):^10}"
                    f" {_ci_fmt(ci_clip, 'AUROC'):^12}"
                    f" {_ci_fmt(ci_clip, 'Sensitivity'):^11}"
                    f" {_ci_fmt(ci_clip, 'Precision'):^11}"
                    f" {'':^8}"
                )
                if ci_line_clip.strip():
                    print(ci_line_clip)
            print("=" * W)

    except Exception as exc:
        print(f"\n  [!] Could not fetch MLflow metrics: {exc}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_experiments(
    task: str,
    plan: list[dict],
    device: torch.device,
    base_config: Config,
    manifest_override: str | None = None,
    dry_run: bool = False,
    num_classes: int | None = None,
    heldout_manifest_path: Path | None = None,
) -> None:
    manifest_path, data_dir = _get_task_paths(task, base_config, manifest_override)
    num_workers = min(base_config.training.num_workers, os.cpu_count() or 8)

    print_plan(task, plan)

    if dry_run:
        print(f"Manifest : {manifest_path}")
        print(f"Data dir : {data_dir}")
        print("Dry-run — no training launched.")
        return

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    pipeline = _build_pipeline(task, base_config, manifest_df, num_classes=num_classes)
    print(f"Classes  : {pipeline.num_classes}  {pipeline.class_names or ''}")

    mlflow.set_tracking_uri(str(base_config.paths.mlflow_db))
    mlflow.set_experiment(pipeline.experiment_name)

    results: list[dict] = []
    total_start = time.time()

    for i, run in enumerate(plan, 1):
        run_label = _fmt_run(i, len(plan), run)
        print(f"\n{'#' * 80}")
        print(f"  {run_label}")
        print(f"{'#' * 80}\n")

        run_config = _apply_run_config(base_config, run)
        pipeline = dc_replace(
            pipeline,
            tune_threshold=run["tune_threshold"],
            tune_clip_threshold=run.get("tune_clip_threshold", False),
        )

        t0 = time.time()
        status = "✓"
        error = ""

        try:
            mlflow.end_run()

            model_entry = MODEL_REGISTRY.get(run_config.model.model)
            if model_entry is None:
                raise ValueError(f"Model '{run_config.model.model}' not in MODEL_REGISTRY")
            img_size = model_entry.img_size

            if run["mode"] == "cv":
                run_cv_mode(
                    run_config,
                    pipeline,
                    device,
                    manifest_path,
                    data_dir,
                    num_workers,
                    img_size,
                    n_splits=run["n_splits"] if run["n_splits"] > 1 else 5,
                    use_full_trainset=False,
                    use_all_splits=True,
                    single_fold=None,
                    register=run["register"],
                    heldout_manifest_path=heldout_manifest_path,
                )
            else:
                run_split_mode(
                    run_config,
                    pipeline,
                    device,
                    manifest_path,
                    data_dir,
                    num_workers,
                    img_size,
                    register=run["register"],
                )

        except KeyboardInterrupt:
            print("\n  [!] Keyboard interrupt — stopping.")
            results.append({**run, "status": "interrupted", "duration_min": 0, "error": ""})
            break

        except Exception as exc:
            status = "✗"
            error = str(exc)
            print(f"\n  [ERROR] run {i} failed: {exc}")
            traceback.print_exc()
            mlflow.end_run()

        finally:
            elapsed = time.time() - t0
            results.append(
                {**run, "status": status, "duration_min": round(elapsed / 60, 1), "error": error}
            )

        print(f"\n  Duration: {timedelta(seconds=int(elapsed))}")

    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'#':<4} {'Model':<25} {'Freeze':<16} {'Status':<5} {'Duration':>8}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        freeze_label = {0: "allBackbone", -1: "freezeBackbone"}.get(
            r["freeze_layers"], f"{r['freeze_layers']}Backbone"
        )
        print(
            f"{i:<4} {r['model']:<25} {freeze_label:<16} "
            f"{r['status']:<5} {r['duration_min']:>6.1f} min"
        )
        if r["error"]:
            print(f"       └─ {r['error'][:80]}")
    n_ok = sum(1 for r in results if r["status"] == "✓")
    n_err = sum(1 for r in results if r["status"] == "✗")
    print("-" * 70)
    print(f"Total : {n_ok} succeeded, {n_err} failed — {timedelta(seconds=int(total_elapsed))}")
    print("=" * 80)

    _print_metrics_report(pipeline, total_start)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment orchestrator for ulcer detection"
    )
    parser.add_argument(
        "--plan",
        default=None,
        help="YAML plan file. If omitted, uses the built-in default plan.",
    )
    parser.add_argument("--model", default=None, help="Filter to a single model from the plan")
    parser.add_argument("--manifest", default=None, help="Override default manifest path")
    parser.add_argument(
        "--heldout-manifest",
        default=None,
        help=(
            "Path to held-out test manifest (CSV). Each fold's model is evaluated on this set. "
            "Use all rows or set split='test' for all rows."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan and exit without training"
    )
    parser.add_argument(
        "--register", action="store_true", help="Register each model in MLflow Model Registry"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        choices=[1, 2],
        default=None,
        help="Override num_classes: 1=sigmoid (default), 2=softmax",
    )
    args = parser.parse_args()

    config = load_config()
    legacy_device = getattr(config, "device", None)
    gpu_id = getattr(legacy_device, "gpu_id", None)
    if gpu_id is None:
        gpu_id = getattr(config.training, "device_id", 0)
    use_cuda = torch.cuda.is_available() and gpu_id >= 0
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
    torch.backends.cudnn.benchmark = True

    if args.plan:
        plan = load_plan(Path(args.plan))
    else:
        plan = build_plan(model_filter=args.model)

    if args.register:
        for r in plan:
            r["register"] = True
    if args.model and args.plan:
        plan = [r for r in plan if r["model"] == args.model]
        if not plan:
            parser.error(f"Model '{args.model}' not found in plan.")

    run_experiments(
        "ulcer",
        plan,
        device,
        config,
        manifest_override=args.manifest,
        dry_run=args.dry_run,
        num_classes=args.num_classes,
        heldout_manifest_path=Path(args.heldout_manifest) if args.heldout_manifest else None,
    )


if __name__ == "__main__":
    main()
