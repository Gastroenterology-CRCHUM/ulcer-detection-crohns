"""
src/evaluation/model_loader.py
-------------------------------
Utilities for instantiating and loading trained ClassifierModel checkpoints.

Public API
~~~~~~~~~~
    remap_classifier_keys   – fix Sequential head key mismatch in old checkpoints
    load_model              – build + load one ClassifierModel from a registry entry
    load_best_models        – load all models in a BEST_MODELS registry dict
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.classifier import ClassifierModel

# ---------------------------------------------------------------------------
# Single-model loader
# ---------------------------------------------------------------------------


def load_model(
    raw_name: str,
    entry: dict,
    project_root: Path,
    *,
    num_classes: int = 1,
    optimizer: str = "AdamW",
    learning_rate: float = 5e-5,
    threshold: float = 0.5,
    dropout_rate: float = 0.5,
    num_epochs: int = 50,
) -> torch.nn.Module:
    """
    Instantiate a ``ClassifierModel`` and load a saved checkpoint.

    Parameters
    ----------
    raw_name      : Registry key, e.g. ``"vitb16_hf-allBackbone"``.
    entry         : Dict with at least ``"path"`` (str relative to project_root)
                    and ``"freeze_backbone"`` (int).
    project_root  : Absolute path to the project root.

    Returns
    -------
    model : ``ClassifierModel`` in eval mode (weights on CPU).

    Warns
    -----
    Prints missing / unexpected keys if any (usually just the classifier head
    when loading DINOv2 / GastroNet self-supervised checkpoints).
    """
    p = Path(entry["path"])
    if not str(p).endswith(".pt"):
        entry["path"] = str(p / "best.pt")
    arch = raw_name.split("-")[0]
    freeze = entry["freeze_backbone"]
    head_type = entry["head_type"]
    checkpoint_path = project_root / entry["path"]

    wrapper = ClassifierModel(
        base_model=arch,
        num_classes=num_classes,
        class_weights=None,
        optimizer=optimizer,
        learning_rate=learning_rate,
        threshold=threshold,
        dropout_rate=dropout_rate,
        num_epochs=num_epochs,
        freeze_layers=freeze,
        head_type=head_type,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    missing, unexpected = wrapper.base_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"  [warn] {raw_name}: {len(missing)} missing key(s)   — {missing[:3]}... , classifier not well initialized"
        )
    if unexpected:
        print(
            f"  [warn] {raw_name}: {len(unexpected)} unexpected key(s) — {unexpected[:3]}... , classifier not well initialized"
        )

    return wrapper


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------


def load_best_models(
    best_models: dict[str, dict],
    project_root: Path,
    **model_kwargs,
) -> None:
    """
    Load all models declared in *best_models* in-place.

    After this call every entry gains a ``"model"`` key holding the loaded
    ``ClassifierModel`` (CPU, eval mode).

    Parameters
    ----------
    best_models   : The ``BEST_MODELS`` registry dict (mutated in place).
    project_root  : Absolute path to the project root.
    **model_kwargs: Forwarded to :func:`load_model` (e.g. ``num_classes=1``).

    Example
    -------
    >>> load_best_models(BEST_MODELS, PROJECT_ROOT)
    >>> model = BEST_MODELS["resnet18-allBackbone"]["model"]
    """
    for raw_name, entry in best_models.items():
        entry["model"] = load_model(raw_name, entry, project_root, **model_kwargs)
        print(f"  ✓  {raw_name}")
