"""
src/training/trainer.py
=======================
High-level training helper used by scripts/train.py.

``load_best_checkpoint`` is kept for the ``train=False`` scenario where
no checkpoint_dir is in scope (the script resumes from a previous run).
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.classifier import ClassifierModel


def load_best_checkpoint(model: ClassifierModel, device: torch.device) -> Path:
    """
    Load the most recent best.pt checkpoint for *model*.

    Used only when ``CONFIG_LAUNCH["train"] = False`` — in all other cases
    ``model.fit()`` returns ``checkpoint_dir`` directly and restores the
    weights before returning, so this function is not needed.

    Args:
        model  : ClassifierModel instance (uses model.name to locate checkpoint).
        device : torch.device.

    Returns:
        Path to the checkpoint directory (used as results_dir downstream).

    Raises:
        FileNotFoundError: If no checkpoint directory is found in any supported
                           checkpoint root for ``{model.name}``.
    """
    candidate_roots = [
        Path("output/ulcer/models/detection"),
        Path("output/ulcer/models"),
        Path("output/models"),  # backward compatibility
    ]

    candidates = []
    searched = []
    for root in candidate_roots:
        base = root / model.name
        searched.append(str(base))
        candidates.extend(base.glob("*/best.pt"))

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in: {', '.join(searched)}")

    checkpoint_path = candidates[-1]
    model.base_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    print(f"  Checkpoint loaded: {checkpoint_path}")
    return checkpoint_path.parent
