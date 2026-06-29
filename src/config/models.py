"""Model registry — backbone definitions for ulcer detection.

How to use
----------
    from src.config.models import MODEL_REGISTRY, get_model_entry

    entry = get_model_entry("vits16_gastronet")
    print(entry.description)   # "ViT-Small/16 — GastroNet-5M / DINOv1"

Paper models (9 configurations)
--------------------------------
These are the models evaluated in the paper. Each key can be passed
directly to the --model flag or used in a YAML plan.

    ResNet-50
      resnet50_imagenet_sup   Supervised ImageNet-1K         (torchvision)
      resnet50_imagenet       Self-sup. DINOv1 / ImageNet    (torch.hub)
      resnet50_gastronet      Self-sup. DINOv1 / GastroNet-5M

    EfficientNet
      efficientnetb0          Supervised ImageNet-1K         (torchvision)

    ViT-Base/16
      vitb16_imagenet_sup     Supervised ImageNet-1K         (torchvision)
      vitb16_imagenet         Self-sup. DINOv1 / ImageNet    (torch.hub)

    ViT-Small/16
      vits16_imagenet_hf      Supervised ImageNet-1K         (timm AugReg)
      vits16_imagenet         Self-sup. DINOv1 / ImageNet    (torch.hub)
      vits16_gastronet        Self-sup. DINOv1 / GastroNet-5M

GastroNet weights
-----------------
Download from the GastroNet-5M paper (Jong et al., Gastroenterology 2026)
and place in the directory configured by cfg.paths.gastronet_weights_dir
(default: data/assets/pretrained/).

    File                          Used by
    RN50_GastroNet-5M_DINOv1.pth resnet50_gastronet
    VITS_GastroNet-5M_DINOv1.pth vits16_gastronet

ImageNet / DINOv1 backbones download automatically via torch.hub or timm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B4_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)


class HeadType(str, Enum):
    """Classification head types."""

    LINEAR = "linear"
    MLP1 = "mlp1"
    MLP2 = "mlp2"


@dataclass
class ModelRegistryEntry:
    """Individual model registry entry."""

    builder: Callable | None
    weights: str | object
    classifier: str
    hub_model: str | None = None
    """Second positional arg to torch.hub.load. None for torchvision models."""
    hf_model_id: str | None = None
    """HuggingFace / timm model ID. When set, builder/hub_model are ignored."""
    gastronet: Path | None = None
    """Path to GastroNet weight file (relative to cfg.paths.gastronet_weights_dir)."""
    img_size: int = 224
    description: str = ""
    architecture: str = ""
    """Human-readable architecture name for display (e.g. 'ViT-Base/16')."""
    pretrain_data: str = ""
    """Pretraining dataset (e.g. 'ImageNet', 'GastroNet-5M')."""
    pretrain_method: str = ""
    """Pretraining method (e.g. 'Supervised', 'Self-sup. (DINOv1)')."""

    def __str__(self) -> str:
        return self.description


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model: str = "resnet50_gastronet"
    """Model key from MODEL_REGISTRY."""

    num_classes: int = 1
    """1 for binary (BCE), 2+ for multi-class (CrossEntropy)."""

    freeze_layers: int = 0
    """
    Layer freezing strategy:
    - 0: full fine-tuning (default)
    - -1: freeze backbone, train head only
    - N: freeze first N transformer blocks / layer groups
    """

    threshold: float = 0.5
    """Binary decision threshold. Ignored for multi-class."""

    dropout_rate: float = 0.5
    """Dropout probability before classification head."""

    head_type: str = HeadType.LINEAR.value
    """Classification head type: linear | mlp1 | mlp2."""

    def __post_init__(self):
        from src.utils import ConfigurationError

        if self.model not in MODEL_REGISTRY:
            raise ConfigurationError(
                f"Model '{self.model}' not in MODEL_REGISTRY.\n"
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        if self.head_type not in [e.value for e in HeadType]:
            raise ConfigurationError(
                f"head_type must be one of {[e.value for e in HeadType]}, got '{self.head_type}'"
            )
        if self.num_classes < 1:
            raise ConfigurationError(f"num_classes must be >= 1, got {self.num_classes}")
        if self.freeze_layers < -1:
            raise ConfigurationError(f"freeze_layers must be >= -1, got {self.freeze_layers}")
        if not 0 < self.threshold < 1:
            raise ConfigurationError(f"threshold must be in (0, 1), got {self.threshold}")


# ── GastroNet weight filenames ────────────────────────────────────────────────
# Paths are relative to cfg.paths.gastronet_weights_dir (default: data/assets/pretrained/)

GASTRONET_WEIGHTS = {
    "resnet50":    Path("RN50_GastroNet-5M_DINOv1.pth"),
    "resnet50_1M": Path("RN50_GastroNet-1M_DINOv1.pth"),
    "resnet50_5M": Path("RN50_GastroNet-5M_DINOv1.pth"),
    "resnet50_200K": Path("RN50_GastroNet-200K_DINOv1.pth"),
    "vits16":      Path("VITS_GastroNet-5M_DINOv1.pth"),
    "vitb16":      Path("VITB16_GastroNet-5M_DINOv1.pth"),
}

_DINO_HUB = "facebookresearch/dino:main"  # torch.hub source for all DINOv1 models


# ── Model Registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {

    # ════════════════════════════════════════════════════════════════
    # ResNet-50  (paper models: resnet50_imagenet_sup, resnet50_imagenet, resnet50_gastronet)
    # ════════════════════════════════════════════════════════════════

    "resnet50_imagenet_sup": ModelRegistryEntry(
        builder=models.resnet50,
        weights=ResNet50_Weights.DEFAULT,
        classifier="fc",
        description="ResNet-50 — ImageNet / Supervised",
        architecture="ResNet-50",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "resnet50_imagenet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=_DINO_HUB,
        hub_model="dino_resnet50",
        classifier="fc",
        description="ResNet-50 — ImageNet / DINOv1",
        architecture="ResNet-50",
        pretrain_data="ImageNet",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_gastronet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=_DINO_HUB,
        hub_model="dino_resnet50",
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50"],
        description="ResNet-50 — GastroNet-5M / DINOv1",
        architecture="ResNet-50",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv1)",
    ),

    # ════════════════════════════════════════════════════════════════
    # EfficientNet  (paper model: efficientnetb0)
    # ════════════════════════════════════════════════════════════════

    "efficientnetb0": ModelRegistryEntry(
        builder=models.efficientnet_b0,
        weights=EfficientNet_B0_Weights.DEFAULT,
        classifier="classifier",
        description="EfficientNet-B0 — ImageNet / Supervised",
        architecture="EfficientNet-B0",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),

    # ════════════════════════════════════════════════════════════════
    # ViT-Base/16  (paper models: vitb16_imagenet_sup, vitb16_imagenet)
    # ════════════════════════════════════════════════════════════════

    "vitb16_imagenet_sup": ModelRegistryEntry(
        builder=models.vit_b_16,
        weights=ViT_B_16_Weights.IMAGENET1K_V1,
        classifier="heads.head",
        description="ViT-Base/16 — ImageNet / Supervised",
        architecture="ViT-Base/16",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "vitb16_imagenet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=_DINO_HUB,
        hub_model="dino_vitb16",
        classifier="head",
        description="ViT-Base/16 — ImageNet / DINOv1",
        architecture="ViT-Base/16",
        pretrain_data="ImageNet",
        pretrain_method="Self-sup. (DINOv1)",
    ),

    # ════════════════════════════════════════════════════════════════
    # ViT-Small/16  (paper models: vits16_imagenet_hf, vits16_imagenet, vits16_gastronet)
    # ════════════════════════════════════════════════════════════════

    "vits16_imagenet_hf": ModelRegistryEntry(
        builder=None,
        weights="timm/vit_small_patch16_224.augreg_in1k",
        classifier="head",
        hf_model_id="timm/vit_small_patch16_224.augreg_in1k",
        description="ViT-Small/16 — ImageNet / Supervised (timm AugReg)",
        architecture="ViT-Small/16",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "vits16_imagenet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=_DINO_HUB,
        hub_model="dino_vits16",
        classifier="head",
        description="ViT-Small/16 — ImageNet / DINOv1",
        architecture="ViT-Small/16",
        pretrain_data="ImageNet",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "vits16_gastronet": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=_DINO_HUB,
        hub_model="dino_vits16",
        classifier="head",
        gastronet=GASTRONET_WEIGHTS["vits16"],
        description="ViT-Small/16 — GastroNet-5M / DINOv1",
        architecture="ViT-Small/16",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv1)",
    ),

    # ════════════════════════════════════════════════════════════════
    # Additional / non-paper models (available but not reported)
    # ════════════════════════════════════════════════════════════════

    "resnet18": ModelRegistryEntry(
        builder=models.resnet18,
        weights=ResNet18_Weights.DEFAULT,
        classifier="fc",
        description="ResNet-18 — ImageNet / Supervised",
        architecture="ResNet-18",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "resnet50_1M": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=_DINO_HUB,
        hub_model="dino_resnet50",
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50_1M"],
        description="ResNet-50 — GastroNet-1M / DINOv1",
        architecture="ResNet-50",
        pretrain_data="GastroNet-1M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_5M": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=_DINO_HUB,
        hub_model="dino_resnet50",
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50_5M"],
        description="ResNet-50 — GastroNet-5M / DINOv1",
        architecture="ResNet-50",
        pretrain_data="GastroNet-5M",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "resnet50_200K": ModelRegistryEntry(
        builder=torch.hub.load,
        weights=_DINO_HUB,
        hub_model="dino_resnet50",
        classifier="fc",
        gastronet=GASTRONET_WEIGHTS["resnet50_200K"],
        description="ResNet-50 — GastroNet-200K / DINOv1",
        architecture="ResNet-50",
        pretrain_data="GastroNet-200K",
        pretrain_method="Self-sup. (DINOv1)",
    ),
    "efficientnetb1": ModelRegistryEntry(
        builder=models.efficientnet_b1,
        weights=EfficientNet_B1_Weights.DEFAULT,
        classifier="classifier",
        img_size=240,
        description="EfficientNet-B1 — ImageNet / Supervised",
        architecture="EfficientNet-B1",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
    "efficientnetb4": ModelRegistryEntry(
        builder=models.efficientnet_b4,
        weights=EfficientNet_B4_Weights.DEFAULT,
        classifier="classifier",
        img_size=380,
        description="EfficientNet-B4 — ImageNet / Supervised",
        architecture="EfficientNet-B4",
        pretrain_data="ImageNet",
        pretrain_method="Supervised",
    ),
}


def get_model_entry(model_name: str) -> ModelRegistryEntry:
    """Get model registry entry by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def get_img_size(model_name: str) -> int:
    """Get input image size for a model."""
    return get_model_entry(model_name).img_size
