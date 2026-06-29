"""
src/evaluation/runner.py
------------------------
Evaluation loop and aggregation helpers for ulcer detection models.

Public API
~~~~~~~~~~
    evaluate_model        – run inference on a DataLoader, return all outputs
    evaluate_all_models   – iterate over a BEST_MODELS registry
    run_delong            – DeLong test at frame or clip level (level=)
"""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation import (
    plot_delong_heatmap,
)
from src.evaluation.aggregation import aggregate_frame_to_clip
from src.evaluation.delong import delong_matrix
from src.models.classifier import ClassifierModel

# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: ClassifierModel,
    dataloader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Run inference on *dataloader* and return all outputs needed downstream.

    The DataLoader must yield ``(images, labels, video_ids, *extras)`` per batch.

    Parameters
    ----------
    model      : A ``ClassifierModel`` (or its ``base_model``).
    dataloader : Test DataLoader.
    device     : ``torch.device`` to run inference on.
    threshold  : Binary classification threshold.

    Returns
    -------
    dict with keys:
        ``labels``        np.ndarray – frame-level ground-truth (0/1)
        ``probs``         np.ndarray – frame-level P(ulcer) in [0, 1]
        ``preds``         np.ndarray – binary predictions at *threshold*
        ``video_ids``     list[str]  – clip ID for each frame
        ``fpr``, ``tpr``  np.ndarray – ROC curve arrays
        ``frame_metrics`` dict       – from :func:`compute_metrics_with_ci`
        ``clip_metrics``  dict       – from :func:`compute_clip_metrics`
    """
    return model.test_evaluation(dataloader, device, threshold)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_all_models(
    best_models: dict[str, dict],
    dataloader,
    device: torch.device,
    display_name_fn,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """
    Evaluate every model in *best_models* and return a results dict keyed by
    display name.

    Parameters
    ----------
    best_models      : ``BEST_MODELS`` registry (each entry must have ``"model"``
                       populated by :func:`~src.evaluation.model_loader.load_best_models`).
    dataloader       : Test DataLoader.
    device           : Inference device.
    display_name_fn  : Callable ``(raw_name: str, *, full: bool) -> str`` for pretty names.
    threshold        : Binary threshold.

    Returns
    -------
    results : ``{display_name: evaluate_model(...)}``
    """
    results: dict[str, dict] = {}

    for raw_name, entry in best_models.items():
        full_name = len(raw_name.split("-")) > 2
        name = display_name_fn(raw_name, full=full_name)
        model = entry["model"]
        print(f"Evaluating: {name} ...")

        results[name] = evaluate_model(model, dataloader, device, threshold)
        results[name]["raw_name"] = raw_name  # for traceability
        results[name]["model"] = model.base_model

        model.to("cpu")
        torch.cuda.empty_cache()

    print("\nEvaluation complete.")
    return results


# ---------------------------------------------------------------------------
# DeLong helpers
# ---------------------------------------------------------------------------


def run_delong(
    results: dict[str, dict],
    *,
    level: str = "frame",
    alpha: float = 0.05,
    save_csv: str | None = None,
):
    """
    Run DeLong pairwise test across all models in *results*.

    Parameters
    ----------
    results  : Output of :func:`evaluate_all_models`.
    level    : ``"frame"`` or ``"clip"``.
    alpha    : Significance level.
    save_csv : Optional path to write the result table as CSV.

    Returns
    -------
    p_matrix   : np.ndarray (n_models, n_models)
    df_delong  : pd.DataFrame with one row per model pair
    fig        : matplotlib Figure (heatmap)
    """
    reference = next(iter(results))
    all_labels = results[reference]["labels"]

    if level == "frame":
        labels_for_test = all_labels
        probs_dict = {name: r["probs"] for name, r in results.items()}
    elif level == "clip":
        clip_probs_dict: dict[str, np.ndarray] = {}
        clip_labels_ref: np.ndarray | None = None

        for name, r in results.items():
            preds = np.asarray(r.get("preds", (np.asarray(r["probs"]) >= 0.5).astype(int)))
            agg_result = aggregate_frame_to_clip(
                np.asarray(r["probs"]),
                preds,
                np.asarray(r["labels"]),
                r["video_ids"],
                method="mean_prob-0.5",
            )
            clip_labels = agg_result["y_true"]
            clip_probs = agg_result["y_prob_clip"]
            clip_probs_dict[name] = clip_probs
            if clip_labels_ref is None:
                clip_labels_ref = clip_labels

        labels_for_test = clip_labels_ref
        probs_dict = clip_probs_dict
    else:
        raise ValueError(f"level must be 'frame' or 'clip', got '{level}'")

    print("\n" + "=" * 60)
    print(f"DELONG TEST — {level.capitalize()}-level AUROC comparisons")
    print("=" * 60)

    p_matrix, df_delong = delong_matrix(
        labels=labels_for_test,
        model_probs=probs_dict,
        alpha=alpha,
    )

    if save_csv:
        df_delong.to_csv(save_csv, index=False)

    print(df_delong.to_string(index=False))
    n_sig = int(df_delong["significant"].sum())
    print(f"\n{n_sig}/{len(df_delong)} significant pairs (α = {alpha})")

    fig = plot_delong_heatmap(p_matrix, df_delong, alpha=alpha)
    return p_matrix, df_delong, fig
