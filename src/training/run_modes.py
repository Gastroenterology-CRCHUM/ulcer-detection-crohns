"""Generic split/CV training runners shared by all pipeline scripts.

Each pipeline (ulcer detection, ulcer size, MES) is described by a
``PipelineDef`` dataclass.  ``run_split_mode`` and ``run_cv_mode`` accept
one of these definitions and delegate all pipeline-specific behaviour to
the fields it carries, keeping the individual script files to ~50 lines.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import mlflow.pytorch as mlflow_pytorch
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient as _MFC
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader as _DataLoader

from src.config import MODEL_REGISTRY, get_img_size
from src.config.loader import config_to_dict
from src.data.dataloader import get_heldout_loader, get_loaders, get_test_loader
from src.evaluation.metrics import compute_metrics_with_ci
from src.evaluation.mlflow_utils import (
    compare_runs_to_markdown,
    get_best_run,
    log_ci_artifact,
    log_confusion_matrix,
    log_dataset_info,
    log_figures,
    log_figures_from_dir,
    log_size_distribution,
    log_split_metrics,
    promote_model,
    register_best_model,
    set_run_tags,
)
from src.evaluation.plots import plot_confusion_matrix_multiclass, plot_roc_curve
from src.evaluation.threshold import collect_probabilities, find_best_threshold, sweep_thresholds
from src.models.classifier import ClassifierModel, _TrainingInterrupted
from src.utils import get_device, set_seed

CV_THRESHOLD = 0.5

warnings.filterwarnings("ignore", message=".*local version label.*")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Pipeline descriptor
# ---------------------------------------------------------------------------


@dataclass
class PipelineDef:
    """Describes a training pipeline for the generic runners.

    Required
    --------
    label_col          : Column name in the manifest that holds the class label.
    num_classes        : Number of output classes (1 for binary, N for multiclass).
    models_root        : Root directory where model checkpoints are saved.
    experiment_name    : MLflow experiment name for this pipeline.
    registry_prefix    : Prefix for MLflow Model Registry names (e.g. "ulcer_").
    run_name_infix     : Infix inserted in the MLflow run name (e.g. "_mes").
    aggregate_by_clip  : Whether to evaluate at clip level after frame inference.
    tune_threshold          : Sweep and select the optimal binary frame threshold on the val set.
    tune_clip_threshold     : Sweep and select the optimal mean-prob clip threshold on the val set.
    is_multiclass           : Use macro averaging and argmax; skip scalar-threshold logic.
    pipeline_tag       : Human-readable tag logged with each MLflow run.

    Optional
    --------
    class_names        : Mapping class_id -> display name (required when is_multiclass).
    comparison_metrics : Metrics forwarded to the run-comparison Markdown table.
    comparison_file_suffix : Suffix appended to the comparison file name.
    use_size_distribution  : Call log_size_distribution instead of log_dataset_info.
    extra_tags         : Extra MLflow tags merged into run tags.
    extra_params       : Extra MLflow params merged into run params.
    """

    label_col: str
    num_classes: int
    models_root: Path
    experiment_name: str
    registry_prefix: str
    run_name_infix: str
    aggregate_by_clip: bool
    tune_threshold: bool
    is_multiclass: bool
    pipeline_tag: str
    tune_clip_threshold: bool = False
    class_names: dict | None = None
    comparison_metrics: list[str] = field(
        default_factory=lambda: ["test__f1_mean", "test__auroc_mean"]
    )
    comparison_file_suffix: str = ""
    use_size_distribution: bool = False
    extra_tags: dict = field(default_factory=dict)
    extra_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_classifier_model(
    cfg, *, pipeline: PipelineDef, threshold: float = 0.5
) -> ClassifierModel:
    """Instantiate a ClassifierModel from project config and pipeline definition."""
    return ClassifierModel(
        base_model=cfg.model.model,
        num_classes=pipeline.num_classes,
        class_weights=cfg.training.class_weights,
        optimizer=cfg.training.optimizer,
        learning_rate=cfg.training.learning_rate,
        threshold=threshold,
        dropout_rate=cfg.training.dropout_rate,
        num_epochs=cfg.training.epochs,
        freeze_layers=cfg.model.freeze_layers,
        gastronet_path=MODEL_REGISTRY[cfg.model.model].gastronet,
        es_patience=cfg.training.es_patience,
        lr_patience=cfg.training.lr_patience,
        lr_factor=cfg.training.lr_factor,
        weight_decay=cfg.training.weight_decay,
        label_smoothing=cfg.training.label_smoothing,
        label_col=pipeline.label_col,
        random_seed=cfg.training.random_seed,
        warmup_epochs=cfg.training.warmup_epochs,
        min_lr=cfg.training.min_lr,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _log_class_weights(model: ClassifierModel, train_df_or_path, label_col: str) -> None:
    weights = model._setup_class_weights(train_df_or_path, label_col=label_col)
    mlflow.log_params({f"class_weight_{i}": round(w, 4) for i, w in enumerate(weights)})


def _log_pytorch_model(model: ClassifierModel, name_suffix: str) -> None:
    try:
        mlflow_pytorch.log_model(
            model.base_model,
            name=f"{model.base_model.__class__.__name__}_{name_suffix}",
        )
    except Exception as exc:
        print(f"  [warn] MLflow model log skipped: {exc}")


def _log_manifest_info(manifest_path: Path, pipeline: PipelineDef) -> None:
    if pipeline.use_size_distribution:
        log_size_distribution(manifest_path, label_col=pipeline.label_col)
    else:
        log_dataset_info(manifest_path)


def _tune_threshold(model: ClassifierModel, best_probs, labels, val_loader, device) -> None:
    if best_probs is None or labels is None:
        try:
            best_probs, labels = collect_probabilities(
                model.base_model, val_loader, device, model.number_classes
            )
        except RuntimeError:
            safe_loader = _DataLoader(
                val_loader.dataset,
                batch_size=val_loader.batch_size,
                num_workers=0,
                pin_memory=False,
            )
            best_probs, labels = collect_probabilities(
                model.base_model, safe_loader, device, model.number_classes
            )
    best_probs = np.array(best_probs)
    if best_probs.ndim == 2:
        best_probs = best_probs[:, 1]
    results = sweep_thresholds(best_probs, labels)
    best = find_best_threshold(results, "f1")
    model.threshold = best["threshold"]
    mlflow.log_metric("best_threshold", model.threshold)
    print(
        f"  Threshold tuning results: {best['f1']:.4f} F1, {best['precision']:.4f} precision, "
        f"{best['recall']:.4f} recall at threshold {best['threshold']:.4f}"
    )


def _tune_clip_threshold(model: ClassifierModel, val_loader, device) -> None:
    """Sweep mean-prob threshold at clip level on the val set and update model.clip_threshold."""
    model.base_model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_clip_ids: list[str] = []

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0], batch[1]
            clip_id = batch[2] if len(batch) > 2 else None
            images = images.to(device)
            logits = model.base_model(images)
            if hasattr(logits, "logits"):
                logits = logits.logits
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            if clip_id is not None:
                all_clip_ids.extend(clip_id)

    if not all_clip_ids or len(all_clip_ids) != len(all_probs):
        print("  [warn] clip_ids not available — clip threshold tuning skipped.")
        return

    clip_data: dict = defaultdict(lambda: {"probs": [], "labels": []})
    for prob, label, cid in zip(all_probs, all_labels, all_clip_ids):
        clip_data[cid]["probs"].append(prob)
        clip_data[cid]["labels"].append(label)

    clip_mean_probs = np.array([np.mean(v["probs"]) for v in clip_data.values()])
    clip_labels = np.array([int(round(np.mean(v["labels"]))) for v in clip_data.values()])

    results = sweep_thresholds(clip_mean_probs, clip_labels)
    best = find_best_threshold(results, "f1")
    model.clip_threshold = best["threshold"]
    mlflow.log_metric("best_clip_threshold", model.clip_threshold)
    print(
        f"  Clip threshold tuning: F1={best['f1']:.4f}  "
        f"precision={best['precision']:.4f}  recall={best['recall']:.4f}  "
        f"threshold={best['threshold']:.4f}  ({len(clip_mean_probs)} clips)"
    )


def _run_test(model: ClassifierModel, test_loader, device, pipeline: PipelineDef) -> dict:
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    return model.test_evaluation(
        test_loader,
        device,
        threshold=model.threshold,
        aggregate_by_clip=pipeline.aggregate_by_clip,
    )


def run_explainability(
    model: ClassifierModel,
    test_loader,
    results: dict,
    device,
    results_dir: Path,
    num_classes: int,
    class_names: dict | None = None,
) -> None:
    """Run Grad-CAM++ or ViT attention visualisation and log artefacts."""
    try:
        from src.visualization.explainability import (
            visualize_test_predictions_with_cam,
            visualize_vit_test_predictions,
        )
    except ModuleNotFoundError as exc:
        print(f"[warn] Explainability skipped (missing dependency): {exc}")
        return

    print("\n" + "=" * 80)
    print("EXPLAINABILITY")
    print("=" * 80)
    save_dir = results_dir / "explainability"
    explainability_path = None

    if "vit" in model.name:
        explainability_path = visualize_vit_test_predictions(
            model=model.unwrapped_backbone,
            test_loader=test_loader,
            test_results=results,
            device=device,
            save_dir=save_dir,
            num_samples=5,
            num_classes=num_classes,
            class_names=class_names,
        )
    else:
        explainability_path = visualize_test_predictions_with_cam(
            model=model.unwrapped_backbone,
            test_loader=test_loader,
            test_results=results,
            model_name=model.name,
            device=device,
            save_dir=save_dir,
            num_samples=5,
            method="gradcam++",
            num_classes=num_classes,
            class_names=class_names,
        )
    if mlflow.active_run() is not None and explainability_path:
        log_figures_from_dir(explainability_path, subdir="explainability")


def _log_test_results(
    test_result: dict,
    model: ClassifierModel,
    results_dir: Path,
    register: bool,
    run_id: str,
    pipeline: PipelineDef,
) -> None:
    """Log frame/clip metrics, artefacts, and optionally register the model."""
    frame_ci = test_result.get("metrics_with_ci", {})
    log_split_metrics(frame_ci, split="test")
    log_ci_artifact(frame_ci, split="test")
    mlflow.log_metric("test_inference_ms", test_result["inference_time_ms"])

    if not pipeline.is_multiclass:
        mlflow.log_metric("test_threshold", test_result["threshold"])
        probs_1d = np.array(test_result.get("probabilities_1d", []))
        labels = np.array(test_result.get("labels", []))
        if probs_1d.ndim == 1 and len(probs_1d) > 0:
            preds_05 = (probs_1d >= 0.5).astype(int)
            _, _, f1_05, _ = precision_recall_fscore_support(
                labels, preds_05, average="binary", zero_division=0
            )
            mlflow.log_metric("test_f1_at05", float(f1_05))

    with tempfile.TemporaryDirectory() as tmp:
        np.save(f"{tmp}/test_probs.npy", test_result.get("probabilities_1d", []))
        np.save(f"{tmp}/test_labels.npy", test_result.get("labels", []))
        mlflow.log_artifact(f"{tmp}/test_probs.npy", "predictions")
        mlflow.log_artifact(f"{tmp}/test_labels.npy", "predictions")

    # Clip-level metrics (binary only)
    if pipeline.aggregate_by_clip and "clip_level" in test_result:
        cl = test_result["clip_level"]
        clip_labels = np.array(cl.get("clip_labels", []))
        clip_preds = np.array(cl.get("clip_predictions", []))
        clip_probs = np.array(cl.get("clip_probs", []))
        if len(clip_labels) > 0 and len(clip_probs) > 0:
            clip_ci = compute_metrics_with_ci(
                clip_labels, clip_preds, clip_probs, seed=model.random_seed
            )
            log_split_metrics(clip_ci, split="test_clip")
            log_ci_artifact(clip_ci, split="test_clip")
        mlflow.log_metric("test_clip_n_clips", cl["n_clips"])
        if "clip_threshold" in test_result:
            mlflow.log_metric("test_clip_threshold", test_result["clip_threshold"])

    # Confusion matrix
    if pipeline.is_multiclass:
        assert pipeline.class_names is not None, (
            "PipelineDef.class_names is required for multiclass"
        )
        cm_fig = plot_confusion_matrix_multiclass(
            cm=test_result["frame_level"]["confusion_matrix"],
            class_names=list(pipeline.class_names.values()),
            title=f"Confusion matrix — {model.name}",
        )
        log_figures({"confusion_matrix": cm_fig}, subdir="test")
    else:
        log_confusion_matrix(
            test_result["frame_level"]["confusion_matrix"],
            test_result["threshold"],
            prefix="test",
        )

    # ROC curve (binary only — multiclass OvR not yet implemented)
    if not pipeline.is_multiclass:
        probs = test_result.get("probabilities_1d", test_result["probabilities"])
        try:
            roc_fig = plot_roc_curve(
                name=model.name,
                labels=test_result["labels"],
                probs=probs,
                threshold=test_result["threshold"],
            )
            log_figures({"roc_curve": roc_fig}, subdir="test")
        except Exception as exc:
            print(f"  [warn] ROC plot skipped: {exc}")

    # Run comparison table
    results_dir.mkdir(parents=True, exist_ok=True)
    compare_runs_to_markdown(
        experiment_name=pipeline.experiment_name,
        metrics=pipeline.comparison_metrics,
        n_runs=10,
        save_path=results_dir / f"runs_comparison{pipeline.comparison_file_suffix}.md",
    )

    # Model Registry
    if register:
        registry_name = f"{pipeline.registry_prefix}{model.name}"
        version = register_best_model(
            run_id=run_id,
            model_name=registry_name,
            description=(
                f"F1={test_result['frame_level']['f1']:.4f}  "
                f"AUROC={test_result['frame_level']['roc_auc']:.4f}"
            ),
        )
        if version:
            best = get_best_run(pipeline.experiment_name, "test__f1_mean")
            if (
                best is None
                or best["metrics"].get("test__f1_mean", 0) <= test_result["frame_level"]["f1"]
            ):
                promote_model(registry_name, version, alias="champion")

    # Checkpoint artefact
    best_pt = results_dir / "best.pt"
    if best_pt.exists():
        mlflow.log_artifact(str(best_pt), artifact_path="checkpoints")
    else:
        print(f"  [warn] best.pt not found in {results_dir}")


def _compute_fold_metrics(
    model: ClassifierModel,
    pipeline: PipelineDef,
    best_probs,
    best_labels,
    val_loader,
    manifest_path: Path,
    device,
    fold: int,
) -> dict:
    """Compute val metrics for one CV fold, handling binary/multiclass averaging.

    Returns a dict with val_f1/precision/recall/auroc plus private _probs/_labels
    for downstream threshold tuning.
    """
    if best_probs is not None and best_labels is not None:
        best_probs = np.array(best_probs)
        best_labels = np.array(best_labels)
        if pipeline.is_multiclass:
            preds = best_probs.argmax(axis=1)
            average = "macro"
        else:
            probs_1d = best_probs[:, 1] if best_probs.ndim == 2 else best_probs
            preds = (probs_1d >= CV_THRESHOLD).astype(int)
            best_probs = probs_1d
            average = "binary"
        precision, recall, f1, _ = precision_recall_fscore_support(
            best_labels, preds, average=average, zero_division=0
        )
        try:
            if pipeline.is_multiclass:
                auroc = float(roc_auc_score(best_labels, best_probs, multi_class="ovr"))
            else:
                auroc = float(roc_auc_score(best_labels, best_probs))
        except ValueError:
            auroc = float("nan")
    else:
        criterion = model._create_criterion(
            model._setup_class_weights(manifest_path, label_col=pipeline.label_col), device
        )
        _, _, precision, recall, f1, auroc, best_probs, best_labels = model.validate(
            val_loader, criterion, device, CV_THRESHOLD
        )
        best_probs = np.array(best_probs)
        if best_probs.ndim == 2:
            best_probs = best_probs[:, 1]

    return {
        "fold": fold,
        "val_f1": f1,
        "val_precision": precision,
        "val_recall": recall,
        "val_auroc": auroc,
        "_probs": best_probs,
        "_labels": best_labels,
    }


# ---------------------------------------------------------------------------
# Split mode
# ---------------------------------------------------------------------------


def run_split_mode(
    cfg,
    pipeline: PipelineDef,
    device,
    manifest_path: Path,
    data_dir: Path,
    num_workers: int,
    img_size: int,
    register: bool,
) -> None:
    pipeline_display = f" — {pipeline.pipeline_tag}" if pipeline.pipeline_tag else ""
    print(f"\n[MODE] Single val-split{pipeline_display}\n")

    set_seed(cfg.training.random_seed)

    train_loader, val_loader = get_loaders(
        mode="split",
        manifest_path=manifest_path,
        data_dir=data_dir,
        batch_size=cfg.training.batch_size,
        img_size=img_size,
        num_workers=num_workers,
        equalize=cfg.training.equalize,
        use_randaugment=cfg.training.use_randaugment,
        randaugment_m=cfg.training.randaugment_m,
        use_random_erasing=cfg.training.use_random_erasing,
        random_erasing_p=cfg.training.random_erasing_p,
        label_col=pipeline.label_col,
    )
    model = build_classifier_model(cfg, pipeline=pipeline)
    infix = pipeline.run_name_infix.lstrip("_")

    with mlflow.start_run(run_name=f"{cfg.model.model}{pipeline.run_name_infix}_split") as run:
        set_run_tags(
            cfg.model.model,
            "split",
            {
                "pipeline": pipeline.pipeline_tag,
                "freeze_layers": cfg.model.freeze_layers,
                **pipeline.extra_tags,
            },
        )
        _log_manifest_info(manifest_path, pipeline)
        mlflow.log_params(
            {
                **config_to_dict(cfg),
                "training_mode": "split",
                **pipeline.extra_params,
            }
        )

        results_dir = pipeline.models_root / model.name
        best_probs = best_labels = None

        if cfg.training.run_train:
            try:
                best_probs, best_labels, results_dir = model.fit(
                    train_loader,
                    val_loader,
                    model.number_epochs,
                    device,
                    manifest_path,
                    checkpoint_root=pipeline.models_root,
                )
            except _TrainingInterrupted as exc:
                print("\n[!] Training interrupted — proceeding to test evaluation.")
                if exc.checkpoint_dir is not None:
                    results_dir = exc.checkpoint_dir
            _log_class_weights(model, train_loader.dataset.df, pipeline.label_col)  # type: ignore[union-attr]
            name_suffix = (
                f"{infix}_{cfg.model.freeze_layers}" if infix else str(cfg.model.freeze_layers)
            )
            _log_pytorch_model(model, name_suffix)

        if pipeline.tune_threshold:
            _tune_threshold(model, best_probs, best_labels, val_loader, device)

        if pipeline.aggregate_by_clip and pipeline.tune_clip_threshold:
            _tune_clip_threshold(model, val_loader, device)

        if cfg.training.run_test:
            test_loader = get_test_loader(
                manifest_path=manifest_path,
                data_dir=data_dir,
                batch_size=cfg.training.batch_size,
                img_size=img_size,
                num_workers=num_workers,
                equalize=cfg.training.equalize,
                label_col=pipeline.label_col,
            )
            test_result = _run_test(model, test_loader, device, pipeline)
            _log_test_results(test_result, model, results_dir, register, run.info.run_id, pipeline)

            if cfg.training.run_explainability:
                run_explainability(
                    model,
                    test_loader,
                    test_result,
                    device,
                    results_dir,
                    pipeline.num_classes,
                    class_names=pipeline.class_names,
                )


# ---------------------------------------------------------------------------
# CV mode
# ---------------------------------------------------------------------------


def run_cv_mode(
    cfg,
    pipeline: PipelineDef,
    device,
    manifest_path: Path,
    data_dir: Path,
    num_workers: int,
    img_size: int,
    n_splits: int,
    use_full_trainset: bool,
    single_fold: int | None,
    register: bool,
    use_all_splits: bool = False,
    heldout_manifest_path: Path | None = None,
    heldout_data_dir: Path | None = None,
) -> None:
    folds_to_run = [single_fold] if single_fold is not None else list(range(n_splits))
    mode_str = f"cv_{n_splits}fold" + ("_fullset" if use_all_splits or use_full_trainset else "")
    pipeline_display = f" — {pipeline.pipeline_tag}" if pipeline.pipeline_tag else ""
    print(
        f"\n[MODE] Cross-validation{pipeline_display}  ({len(folds_to_run)} fold(s), {mode_str})\n"
    )

    fold_metrics: list[dict] = []
    fold_checkpoint_dirs: list[Path | None] = []
    infix = pipeline.run_name_infix.lstrip("_")

    with mlflow.start_run(
        run_name=f"{cfg.model.model}{pipeline.run_name_infix}_{mode_str}"
    ) as parent_run:
        set_run_tags(
            cfg.model.model,
            mode_str,
            {
                "pipeline": pipeline.pipeline_tag,
                "n_splits": n_splits,
                **pipeline.extra_tags,
            },
        )
        _log_manifest_info(manifest_path, pipeline)
        mlflow.log_params(
            {
                **config_to_dict(cfg),
                "training_mode": mode_str,
                "n_splits": n_splits,
                **pipeline.extra_params,
            }
        )

        heldout_loader = None
        if heldout_manifest_path is not None:
            heldout_loader = get_heldout_loader(
                manifest_path=heldout_manifest_path,
                data_dir=heldout_data_dir or data_dir,
                batch_size=cfg.training.batch_size,
                img_size=img_size,
                num_workers=num_workers,
                equalize=cfg.training.equalize,
                label_col=pipeline.label_col,
            )
            mlflow.log_param("heldout_manifest", heldout_manifest_path.name)
            print(f"  Heldout test set loaded  ({heldout_manifest_path.name})\n")

        for fold in folds_to_run:
            print(f"\n{'=' * 80}\n  FOLD {fold + 1} / {n_splits}\n{'=' * 80}")

            set_seed(cfg.training.random_seed + fold)

            train_loader, val_loader = get_loaders(
                mode="cv",
                manifest_path=manifest_path,
                data_dir=data_dir,
                batch_size=cfg.training.batch_size,
                img_size=img_size,
                fold=fold,
                n_splits=n_splits,
                use_full_trainset=use_full_trainset,
                use_all_splits=use_all_splits,
                num_workers=num_workers,
                equalize=cfg.training.equalize,
                use_randaugment=cfg.training.use_randaugment,
                randaugment_m=cfg.training.randaugment_m,
                use_random_erasing=cfg.training.use_random_erasing,
                random_erasing_p=cfg.training.random_erasing_p,
                random_seed=cfg.training.random_seed,
                label_col=pipeline.label_col,
            )
            model = build_classifier_model(cfg, pipeline=pipeline)

            with mlflow.start_run(run_name=f"fold_{fold + 1}", nested=True):
                mlflow.log_param("fold", fold)

                best_probs = best_labels = None
                fold_ckpt_dir: Path | None = None
                if cfg.training.run_train:
                    print("\n" + "=" * 80)
                    print("TRAINING")
                    print("=" * 80)
                    best_probs, best_labels, fold_ckpt_dir = model.fit(
                        train_loader,
                        val_loader,
                        model.number_epochs,
                        device,
                        manifest_path,
                        checkpoint_root=pipeline.models_root,
                    )
                    _log_class_weights(model, train_loader.dataset.df, pipeline.label_col)  # type: ignore[union-attr]
                    name_suffix = (
                        f"{infix}_{cfg.model.freeze_layers}"
                        if infix
                        else str(cfg.model.freeze_layers)
                    )
                    _log_pytorch_model(model, name_suffix)
                    if fold_ckpt_dir is not None:
                        fold_best_pt = fold_ckpt_dir / "best.pt"
                        if fold_best_pt.exists():
                            mlflow.log_artifact(str(fold_best_pt), artifact_path="checkpoints")

                fm = _compute_fold_metrics(
                    model,
                    pipeline,
                    best_probs,
                    best_labels,
                    val_loader,
                    manifest_path,
                    device,
                    fold,
                )

                # Log per-fold val probabilities as artifacts
                with tempfile.TemporaryDirectory() as _tmp:
                    np.save(f"{_tmp}/val_probs_fold{fold}.npy", fm["_probs"])
                    np.save(f"{_tmp}/val_labels_fold{fold}.npy", fm["_labels"])
                    mlflow.log_artifact(f"{_tmp}/val_probs_fold{fold}.npy", "predictions/cv_folds")
                    mlflow.log_artifact(f"{_tmp}/val_labels_fold{fold}.npy", "predictions/cv_folds")

                # Threshold tuning per fold (binary only)
                fold_probs = fm.pop("_probs", None)
                fold_labels = fm.pop("_labels", None)
                if pipeline.tune_threshold and fold_probs is not None and fold_labels is not None:
                    _tune_threshold(model, fold_probs, fold_labels, val_loader, device)
                    optimal_threshold = model.threshold
                    # Recompute F1/P/R at tuned threshold so fold summary is consistent
                    preds = (fold_probs >= optimal_threshold).astype(int)
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        fold_labels, preds, average="binary", zero_division=0
                    )
                    fm["val_f1"] = float(f1)
                    fm["val_precision"] = float(prec)
                    fm["val_recall"] = float(rec)
                else:
                    optimal_threshold = CV_THRESHOLD

                if pipeline.aggregate_by_clip and pipeline.tune_clip_threshold:
                    _tune_clip_threshold(model, val_loader, device)
                    optimal_clip_threshold = model.clip_threshold
                else:
                    optimal_clip_threshold = model.clip_threshold

                fold_entry = {k: v for k, v in fm.items() if not k.startswith("_")}
                if pipeline.tune_threshold:
                    fold_entry["optimal_threshold"] = optimal_threshold
                if pipeline.aggregate_by_clip and pipeline.tune_clip_threshold:
                    fold_entry["optimal_clip_threshold"] = optimal_clip_threshold

                if heldout_loader is not None:
                    _heldout = model.test_evaluation(
                        heldout_loader,
                        device,
                        threshold=optimal_threshold,
                        aggregate_by_clip=pipeline.aggregate_by_clip,
                    )
                    log_split_metrics(_heldout.get("metrics_with_ci", {}), split="heldout")
                    log_ci_artifact(_heldout.get("metrics_with_ci", {}), split="heldout")
                    fold_entry["heldout_f1"] = _heldout["frame_level"]["f1"]
                    fold_entry["heldout_auroc"] = _heldout["frame_level"]["roc_auc"]
                    with tempfile.TemporaryDirectory() as _tmp:
                        np.save(
                            f"{_tmp}/heldout_probs_fold{fold}.npy",
                            _heldout.get("probabilities_1d", []),
                        )
                        np.save(
                            f"{_tmp}/heldout_labels_fold{fold}.npy",
                            _heldout.get("labels", []),
                        )
                        mlflow.log_artifact(
                            f"{_tmp}/heldout_probs_fold{fold}.npy", "predictions/heldout"
                        )
                        mlflow.log_artifact(
                            f"{_tmp}/heldout_labels_fold{fold}.npy", "predictions/heldout"
                        )
                    print(
                        f"  Heldout    F1={_heldout['frame_level']['f1']:.4f}  "
                        f"AUROC={_heldout['frame_level']['roc_auc']:.4f}"
                    )

                fold_metrics.append(fold_entry)
                fold_checkpoint_dirs.append(fold_ckpt_dir)

                log_dict: dict = {
                    "fold_val_f1": float(fm["val_f1"]),
                    "fold_val_precision": float(fm["val_precision"]),
                    "fold_val_recall": float(fm["val_recall"]),
                    "fold_val_auroc": float(fm["val_auroc"]),
                }
                if pipeline.tune_threshold:
                    log_dict["fold_optimal_threshold"] = float(optimal_threshold)
                if pipeline.aggregate_by_clip and pipeline.tune_clip_threshold:
                    log_dict["fold_optimal_clip_threshold"] = float(optimal_clip_threshold)
                if "heldout_auroc" in fold_entry:
                    log_dict["fold_heldout_f1"] = float(fold_entry["heldout_f1"])
                    log_dict["fold_heldout_auroc"] = float(fold_entry["heldout_auroc"])
                mlflow.log_metrics(log_dict, step=fold)

                suffix = (
                    f"  (thr={optimal_threshold:.3f})" if pipeline.tune_threshold else "  (macro)"
                )
                print(
                    f"  Fold {fold} → F1={fm['val_f1']:.4f}  "
                    f"P={fm['val_precision']:.4f}  R={fm['val_recall']:.4f}  "
                    f"AUROC={fm['val_auroc']:.4f}{suffix}"
                )

        # ── CV Summary ───────────────────────────────────────────────────────
        metrics_df = pd.DataFrame(fold_metrics)
        print("\n" + "=" * 80)
        print(f"CROSS-VALIDATION SUMMARY{pipeline_display}")
        print("=" * 80)
        print(metrics_df.to_string(index=False))

        for col in ("val_f1", "val_precision", "val_recall", "val_auroc"):
            col_data = metrics_df[col].dropna()
            mean, std = col_data.mean(), col_data.std()
            print(f"  {col:20s}  {mean:.4f} ± {std:.4f}")
            mlflow.log_metrics({f"cv_mean_{col}": mean, f"cv_std_{col}": std})

        # Threshold dispersion (binary only) — informative, not used for evaluation.
        if pipeline.tune_threshold and "optimal_threshold" in metrics_df.columns:
            opt_thr = metrics_df["optimal_threshold"]
            thr_range = f"[{opt_thr.min():.3f}–{opt_thr.max():.3f}]"
            print(
                f"\n  Threshold  mean={opt_thr.mean():.4f}  std={opt_thr.std():.4f}  range {thr_range}"
            )
            mlflow.log_metrics(
                {
                    "cv_mean_threshold": float(opt_thr.mean()),
                    "cv_std_threshold": float(opt_thr.std()),
                    "cv_min_threshold": float(opt_thr.min()),
                    "cv_max_threshold": float(opt_thr.max()),
                }
            )

        if (
            pipeline.aggregate_by_clip
            and pipeline.tune_clip_threshold
            and "optimal_clip_threshold" in metrics_df.columns
        ):
            clip_thr = metrics_df["optimal_clip_threshold"]
            print(
                f"  Clip thr   mean={clip_thr.mean():.4f}  std={clip_thr.std():.4f}  "
                f"range [{clip_thr.min():.3f}–{clip_thr.max():.3f}]"
            )
            mlflow.log_metrics(
                {
                    "cv_mean_clip_threshold": float(clip_thr.mean()),
                    "cv_std_clip_threshold": float(clip_thr.std()),
                    "cv_min_clip_threshold": float(clip_thr.min()),
                    "cv_max_clip_threshold": float(clip_thr.max()),
                }
            )

        if heldout_loader is not None and "heldout_auroc" in metrics_df.columns:
            print("\n  ── Heldout test-set performance (per fold) ──────────────────────────────")
            for _col in ("heldout_f1", "heldout_auroc"):
                if _col not in metrics_df.columns:
                    continue
                _col_data = metrics_df[_col].dropna()
                _mean, _std = _col_data.mean(), _col_data.std()
                print(f"  {_col:24s}  {_mean:.4f} ± {_std:.4f}")
                mlflow.log_metrics({f"cv_mean_{_col}": _mean, f"cv_std_{_col}": _std})

        # ── Best-fold evaluation ──────────────────────────────────────────────
        # Select the fold with the highest val AUROC.
        # Reload its checkpoint, rebuild its val loader, run inference.
        # Log full CI metrics, confusion matrix, ROC curve, explainability,
        # and the checkpoint itself on the parent run.
        best_list_idx = int(np.argmax(metrics_df["val_auroc"].to_numpy(dtype=float, na_value=0.0)))
        best_fold_row = metrics_df.iloc[best_list_idx]
        best_fold = int(best_fold_row["fold"])
        best_threshold = float(best_fold_row.get("optimal_threshold", CV_THRESHOLD))
        best_clip_threshold = float(best_fold_row.get("optimal_clip_threshold", best_threshold))
        best_ckpt_dir = fold_checkpoint_dirs[best_list_idx]

        print(
            f"\n  Best fold : {best_fold + 1}  "
            f"(val AUROC={best_fold_row['val_auroc']:.4f}, thr={best_threshold:.3f})"
        )
        mlflow.log_params(
            {
                "best_fold": best_fold,
                "best_fold_val_auroc": float(best_fold_row["val_auroc"]),
                "best_fold_threshold": best_threshold,
            }
        )

        best_ckpt_path = best_ckpt_dir / "best.pt" if best_ckpt_dir else None
        if best_ckpt_path and best_ckpt_path.exists():
            # Log checkpoint artifact on parent run
            mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")

            # Reload best fold model
            best_model = build_classifier_model(cfg, pipeline=pipeline, threshold=best_threshold)
            best_model.base_model.load_state_dict(
                torch.load(best_ckpt_path, map_location=device, weights_only=True)
            )
            best_model.base_model.to(device)
            best_model.threshold = best_threshold
            if pipeline.aggregate_by_clip:
                best_model.clip_threshold = best_clip_threshold

            # Rebuild val loader for the best fold (deterministic given same seed)
            _, best_val_loader = get_loaders(
                mode="cv",
                manifest_path=manifest_path,
                data_dir=data_dir,
                batch_size=cfg.training.batch_size,
                img_size=img_size,
                fold=best_fold,
                n_splits=n_splits,
                use_full_trainset=use_full_trainset,
                use_all_splits=use_all_splits,
                num_workers=num_workers,
                equalize=cfg.training.equalize,
                random_seed=cfg.training.random_seed,
                label_col=pipeline.label_col,
            )

            print("\n" + "=" * 80)
            print(f"BEST FOLD ({best_fold + 1}) EVALUATION")
            print("=" * 80)
            best_result = best_model.test_evaluation(
                best_val_loader,
                device,
                threshold=best_threshold,
                aggregate_by_clip=pipeline.aggregate_by_clip,
            )

            # CI metrics
            best_ci = best_result.get("metrics_with_ci", {})
            log_split_metrics(best_ci, split="best_fold")
            log_ci_artifact(best_ci, split="best_fold")
            mlflow.log_metric("best_fold_threshold", best_threshold)

            # Save best-fold probs / labels for DeLong in notebook
            with tempfile.TemporaryDirectory() as _tmp:
                np.save(f"{_tmp}/best_fold_probs.npy", best_result.get("probabilities_1d", []))
                np.save(f"{_tmp}/best_fold_labels.npy", best_result.get("labels", []))
                mlflow.log_artifact(f"{_tmp}/best_fold_probs.npy", "predictions")
                mlflow.log_artifact(f"{_tmp}/best_fold_labels.npy", "predictions")

            # Confusion matrix
            if pipeline.is_multiclass:
                assert pipeline.class_names is not None
                cm_fig = plot_confusion_matrix_multiclass(
                    cm=best_result["frame_level"]["confusion_matrix"],
                    class_names=list(pipeline.class_names.values()),
                    title=f"Confusion matrix — {best_model.name} (best fold {best_fold + 1})",
                )
                log_figures({"confusion_matrix": cm_fig}, subdir="best_fold")
            else:
                log_confusion_matrix(
                    best_result["frame_level"]["confusion_matrix"],
                    best_threshold,
                    prefix="best_fold",
                )

            # ROC curve
            if not pipeline.is_multiclass:
                probs = best_result.get("probabilities_1d", best_result["probabilities"])
                try:
                    roc_fig = plot_roc_curve(
                        name=best_model.name,
                        labels=best_result["labels"],
                        probs=probs,
                        threshold=best_threshold,
                    )
                    log_figures({"roc_curve": roc_fig}, subdir="best_fold")
                except Exception as exc:
                    print(f"  [warn] ROC plot skipped: {exc}")

            # Clip-level metrics on best fold
            if pipeline.aggregate_by_clip and "clip_level" in best_result:
                cl = best_result["clip_level"]
                clip_labels = np.array(cl.get("clip_labels", []))
                clip_preds = np.array(cl.get("clip_predictions", []))
                clip_probs = np.array(cl.get("clip_probs", []))
                if len(clip_labels) > 0 and len(clip_probs) > 0:
                    clip_ci = compute_metrics_with_ci(
                        clip_labels, clip_preds, clip_probs, seed=best_model.random_seed
                    )
                    log_split_metrics(clip_ci, split="best_fold_clip")
                    log_ci_artifact(clip_ci, split="best_fold_clip")
                mlflow.log_metric("best_fold_clip_n_clips", cl["n_clips"])

            # Held-out test set evaluation for the best fold
            if heldout_loader is not None:
                print("\n" + "=" * 80)
                print(f"BEST FOLD ({best_fold + 1}) — HELDOUT TEST SET")
                print("=" * 80)
                _heldout_best = best_model.test_evaluation(
                    heldout_loader,
                    device,
                    threshold=best_threshold,
                    aggregate_by_clip=pipeline.aggregate_by_clip,
                )
                log_split_metrics(_heldout_best.get("metrics_with_ci", {}), split="heldout")
                log_ci_artifact(_heldout_best.get("metrics_with_ci", {}), split="heldout")
                with tempfile.TemporaryDirectory() as _tmp:
                    np.save(
                        f"{_tmp}/heldout_best_fold_probs.npy",
                        _heldout_best.get("probabilities_1d", []),
                    )
                    np.save(
                        f"{_tmp}/heldout_best_fold_labels.npy",
                        _heldout_best.get("labels", []),
                    )
                    mlflow.log_artifact(f"{_tmp}/heldout_best_fold_probs.npy", "predictions")
                    mlflow.log_artifact(f"{_tmp}/heldout_best_fold_labels.npy", "predictions")
                if not pipeline.is_multiclass:
                    _probs = _heldout_best.get(
                        "probabilities_1d", _heldout_best["probabilities"]
                    )
                    try:
                        _roc_fig = plot_roc_curve(
                            name=best_model.name,
                            labels=_heldout_best["labels"],
                            probs=_probs,
                            threshold=best_threshold,
                        )
                        log_figures({"roc_curve": _roc_fig}, subdir="heldout")
                    except Exception as exc:
                        print(f"  [warn] Heldout ROC plot skipped: {exc}")
                print(
                    f"  F1={_heldout_best['frame_level']['f1']:.4f}  "
                    f"AUROC={_heldout_best['frame_level']['roc_auc']:.4f}"
                )

            # Explainability
            if cfg.training.run_explainability:
                run_explainability(
                    best_model,
                    best_val_loader,
                    best_result,
                    device,
                    best_ckpt_dir,  # type: ignore[arg-type]  # narrowed by outer `if best_ckpt_path.exists()`
                    pipeline.num_classes,
                    class_names=pipeline.class_names,
                )

            # Model registry
            if register:
                registry_name = f"{pipeline.registry_prefix}{best_model.name}"
                version = register_best_model(
                    run_id=parent_run.info.run_id,
                    model_name=registry_name,
                    description=(
                        f"CV best fold {best_fold + 1}  "
                        f"F1={best_result['frame_level']['f1']:.4f}  "
                        f"AUROC={best_result['frame_level']['roc_auc']:.4f}"
                    ),
                )
                if version:
                    best_registered = get_best_run(pipeline.experiment_name, "cv_mean_val_auroc")
                    if best_registered is None or best_registered["metrics"].get(
                        "cv_mean_val_auroc", 0
                    ) <= float(metrics_df["val_auroc"].mean()):
                        promote_model(registry_name, version, alias="champion")

            # Run comparison table — models_root / model_name (3 levels up from best.pt)
            results_dir = best_ckpt_dir.parent.parent.parent  # type: ignore[union-attr]
            results_dir.mkdir(parents=True, exist_ok=True)
            compare_runs_to_markdown(
                experiment_name=pipeline.experiment_name,
                metrics=["cv_mean_val_auroc", "cv_mean_val_f1"],
                n_runs=10,
                save_path=results_dir / f"runs_comparison{pipeline.comparison_file_suffix}.md",
            )
        else:
            print("  [warn] Best fold checkpoint not found — skipping best-fold evaluation.")

    mlflow.end_run()


# ---------------------------------------------------------------------------
# Ensemble inference (external test set)
# ---------------------------------------------------------------------------


def run_ensemble_inference(
    parent_run_id: str,
    cfg,
    pipeline: PipelineDef,
    device,
    manifest_path: Path,
    data_dir: Path,
    num_workers: int,
    img_size: int,
    cv_threshold: float = 0.5,
) -> dict:
    """Average predictions of all CV fold checkpoints on a held-out test set.

    Each fold's best.pt (logged as a child-run artifact) is loaded in turn,
    run on the test loader, and the resulting probabilities are averaged.
    Threshold defaults to 0.5; pass cv_mean_threshold from the parent run for
    threshold-consistent reporting.
    """
    _client = _MFC()
    parent_run = _client.get_run(parent_run_id)
    experiment_id = parent_run.info.experiment_id

    child_runs = _client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    if not child_runs:
        raise ValueError(f"No child runs found for parent run {parent_run_id}")

    child_runs = sorted(child_runs, key=lambda r: int(r.data.params.get("fold", 0)))

    test_loader = get_test_loader(
        manifest_path=manifest_path,
        data_dir=data_dir,
        batch_size=cfg.training.batch_size,
        img_size=img_size,
        num_workers=num_workers,
        equalize=cfg.training.equalize,
        label_col=pipeline.label_col,
    )

    all_probs: list[np.ndarray] = []
    labels_arr: np.ndarray | None = None
    loaded_folds: list[int] = []

    for child in child_runs:
        fold = int(child.data.params.get("fold", -1))
        child_id = child.info.run_id
        try:
            ckpt_path = _client.download_artifacts(child_id, "checkpoints/best.pt")
        except Exception:
            print(f"  [warn] Fold {fold + 1}: checkpoint not found in MLflow, skipping.")
            continue

        model = build_classifier_model(cfg, pipeline=pipeline, threshold=cv_threshold)
        model.base_model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.base_model.to(device)

        probs, labels = collect_probabilities(
            model.base_model, test_loader, device, model.number_classes
        )
        probs = np.array(probs)
        if probs.ndim == 2:
            probs = probs[:, 1]
        all_probs.append(probs)
        if labels_arr is None:
            labels_arr = np.array(labels)
        loaded_folds.append(fold)

        try:
            fold_auroc = float(roc_auc_score(labels_arr, probs))
        except ValueError:
            fold_auroc = float("nan")
        print(f"  Fold {fold + 1}: {len(probs)} frames, AUROC={fold_auroc:.4f}")

    if not all_probs or labels_arr is None:
        raise ValueError("No fold checkpoints could be loaded.")

    ensemble_probs = np.mean(all_probs, axis=0)
    try:
        auroc = float(roc_auc_score(labels_arr, ensemble_probs))
    except ValueError:
        auroc = float("nan")

    preds = (ensemble_probs >= cv_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_arr, preds, average="binary", zero_division=0
    )

    print(
        f"\n  Ensemble ({len(loaded_folds)} folds): AUROC={auroc:.4f}  F1={f1:.4f}  "
        f"Sens={recall:.4f}  Prec={precision:.4f}  thr={cv_threshold:.3f}"
    )

    return {
        "ensemble_probs": ensemble_probs,
        "labels": labels_arr,
        "loaded_folds": loaded_folds,
        "auroc": auroc,
        "f1": float(f1),
        "sensitivity": float(recall),
        "precision": float(precision),
        "threshold": cv_threshold,
    }


# ---------------------------------------------------------------------------
# CLI setup helper
# ---------------------------------------------------------------------------


def setup_training(cfg, *, description: str = "Train classifier"):
    """Parse standard CLI arguments and configure CUDA.

    Returns (args, device, img_size, num_workers).
    The returned ``args`` also exposes ``args.manifest`` for manifest overrides.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["cv", "split"], default="split")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=cfg.cv.n_splits)
    parser.add_argument(
        "--use-full-trainset", action="store_true", default=cfg.cv.use_full_trainset
    )
    parser.add_argument(
        "--register", action="store_true", help="Register model in MLflow Model Registry"
    )
    parser.add_argument("--manifest", type=str, default=None, help="Override default manifest path")
    args = parser.parse_args()

    device = get_device()
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(cfg.training.device_id))
    torch.backends.cudnn.benchmark = True

    img_size = get_img_size(cfg.model.model)
    num_workers = min(cfg.training.num_workers, os.cpu_count() or 8)

    return args, device, img_size, num_workers
