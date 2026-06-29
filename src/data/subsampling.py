"""Visual diversity subsampling using GastroNet backbone embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_backbone_for_embeddings(
    arch: str = "resnet50_gastronet",
    checkpoint_path: Path | None = None,
    *,
    device: str = "cpu",
) -> nn.Module | None:
    """Build a GastroNet backbone for embedding extraction.

    Loads pretrained GastroNet weights from the model registry by default.
    If checkpoint_path is provided and exists, its weights override the registry weights.
    Returns None on any import or loading error (callers fall back to uniform stride).
    """
    try:
        from src.config import get_model_entry
        from src.models.classifier import ClassifierModel
    except ImportError as exc:
        logger.warning("Cannot import ClassifierModel (%s) — uniform stride fallback.", exc)
        return None

    try:
        entry = get_model_entry(arch)
        model = ClassifierModel(
            base_model=arch,
            num_classes=1,
            class_weights=None,
            optimizer="AdamW",
            learning_rate=1e-4,
            threshold=0.5,
            dropout_rate=0.0,
            num_epochs=1,
            freeze_layers=-1,
            gastronet_path=entry.gastronet,
        )
    except Exception as exc:
        logger.warning("Failed to build backbone (%s) — uniform stride fallback.", exc)
        return None

    if checkpoint_path is not None and checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            state_dict = (
                state_dict.get("state_dict") or state_dict.get("model_state_dict") or state_dict
            )
        model.base_model.load_state_dict(state_dict, strict=False)
        logger.info("Backbone weights overridden from %s.", checkpoint_path.name)

    # Strip classification head → return raw feature vectors
    parts = entry.classifier.split(".")
    obj = model.base_model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], nn.Identity())

    backbone = model.base_model.eval().to(device)
    logger.info("Backbone ready (arch=%s, device=%s).", arch, device)
    return backbone


def _extract_embeddings(
    backbone: nn.Module,
    frame_paths: list[Path],
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    embeddings_list: list[np.ndarray] = []
    backbone.eval()
    with torch.no_grad():
        for i in range(0, len(frame_paths), batch_size):
            tensors = []
            for p in frame_paths[i : i + batch_size]:
                img = cv2.imread(str(p))
                if img is None:
                    tensors.append(torch.zeros(3, 224, 224))
                    continue
                tensors.append(_TRANSFORM(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            batch = torch.stack(tensors).to(device)
            out = backbone(batch).flatten(1)
            embeddings_list.append(out.cpu().numpy())
    return np.vstack(embeddings_list)


def visual_subsample(
    frame_paths: list[Path],
    max_count: int,
    *,
    backbone: nn.Module | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> list[Path]:
    """Select up to max_count frames maximising visual diversity.

    Uses greedy farthest-point sampling on backbone embeddings when a backbone
    is provided; falls back to uniform stride otherwise.
    """
    if len(frame_paths) <= max_count:
        return list(frame_paths)

    if backbone is None:
        indices = np.round(np.linspace(0, len(frame_paths) - 1, max_count)).astype(int)
        return [frame_paths[i] for i in indices]

    logger.info("Extracting embeddings for %d frames …", len(frame_paths))
    embeddings = _extract_embeddings(backbone, frame_paths, device=device, batch_size=batch_size)

    # Greedy farthest-point sampling (maximise minimum pairwise distance)
    selected = [0]
    min_dists = np.linalg.norm(embeddings - embeddings[0], axis=1).astype(np.float64)
    min_dists[0] = 0.0

    for _ in range(max_count - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        d = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
        np.minimum(min_dists, d, out=min_dists)
        min_dists[next_idx] = 0.0

    logger.info("Selected %d/%d frames via farthest-point sampling.", max_count, len(frame_paths))
    return [frame_paths[i] for i in sorted(selected)]
