"""
src/evaluation/style.py
------------------------
Shared presentation layer for ulcer-detection comparison figures (effect
decomposition, CV results, ...): color and display text.
Pure constants/helpers, reused by every script that plots or tabulates
per-config comparisons so a config's color and
name always render the same way across every figure in the manuscript.

Neither is ever safe to use as a join key, dict key, DataFrame index, or
filename. config_color/config_label raise on any string that is not one
of the 9 MODEL_REGISTRY keys, exactly like get_model_entry().

Public API
----------
config_color(model_key) -> hex str
    Consistent color for one of the 9 MODEL_REGISTRY configs.
config_label(model_key, *, short=True) -> str
    Display name. short=True: compact, for legends/axis ticks/heatmap rows
    (<=21 chars, e.g. "RN50 GN-5M DINOv1"). short=False: full form for table
    cells and single-config titles (e.g. "ResNet-50, GastroNet-5M / Self-sup.
    (DINOv1)").
config_labels(model_keys, *, short=True) -> list[str]
    Vectorised config_label.
metric_label(metric) -> str
    Display name for a lowercase metric key ("auroc" -> "AUROC"). Passes
    unknown metrics through unchanged (non-strict, unlike config_label).
level_label(level) -> str
    "frame" -> "Frame-level", "clip" -> "Clip-level".
slug(text) -> str
    Filename-safe fragment (e.g. "ViT-Base/16" -> "vitbase16"). Output is an
    on-disk filename contract (arch_<slug>_*.png) -- do not change behavior.
shade(hex_color, amount) -> hex str
    Lighten (amount > 0) or darken (amount < 0) a hex color.
ARCH_HUE            - dict[architecture -> base hex]
STATUS_SIGNIFICANT  - green, "this test/CI is significant"
STATUS_NOT_SIGNIFICANT - pale red, "this test/CI is not significant"
STATUS_NEUTRAL      - muted gray ink, non-committal annotations/captions
"""

from __future__ import annotations

from collections.abc import Iterable

from src.config.models import get_model_entry

ARCH_HUE = {
    "ResNet-50": "#2a78d6",
    "EfficientNet-B0": "#eb6834",
    "ViT-Base/16": "#1baf7a",
    "ViT-Small/16": "#eda100",
}

# Abbreviated field values for the SHORT display form.
# Keep ARCH_SHORT's key set identical to ARCH_HUE's.
ARCH_SHORT = {
    "ResNet-50": "RN50",
    "EfficientNet-B0": "EffNet-B0",
    "ViT-Base/16": "ViT-B/16",
    "ViT-Small/16": "ViT-S/16",
}
CORPUS_SHORT = {"ImageNet-1K": "IN-1K", "GastroNet-5M": "GN-5M"}
METHOD_SHORT = {"Supervised": "Sup.", "Self-sup. (DINOv1)": "DINOv1"}

METRIC_LABEL = {
    "auroc": "AUROC",
    "sensitivity": "Sensitivity",
    "recall": "Sensitivity",  # run_modes.py logs "recall"; same quantity as sensitivity
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1",
    "accuracy": "Accuracy",
    "threshold": "Threshold",
}

LEVEL_LABEL = {"frame": "Frame-level", "clip": "Clip-level"}

STATUS_SIGNIFICANT = "#0ca30c"
STATUS_NOT_SIGNIFICANT = "#f5cfc4"
STATUS_NEUTRAL = "#898781"


def shade(hex_color: str, amount: float) -> str:
    """Lighten (amount > 0) or darken (amount < 0) a hex color toward white/black."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    if amount >= 0:
        r, g, b = (r + (255 - r) * amount, g + (255 - g) * amount, b + (255 - b) * amount)
    else:
        r, g, b = (r * (1 + amount), g * (1 + amount), b * (1 + amount))
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def _corpus(entry) -> str:
    """Canonical pretraining corpus"""
    return "GastroNet-5M" if entry.pretrain_data.startswith("GastroNet") else "ImageNet-1K"


def _pretrain_key(entry) -> str:
    """'supervised' | 'gastronet_ssl' | 'imagenet_ssl' -- the one 3-way
    partition that is SIMULTANEOUSLY config_color()'s shade branch and
    config_label()'s (corpus, method) token pair, so a color and its label
    can never desync: both call this, neither re-derives it."""
    if entry.pretrain_method == "Supervised":
        return "supervised"
    if entry.pretrain_data.startswith("GastroNet"):
        return "gastronet_ssl"
    return "imagenet_ssl"


def config_color(model_key: str) -> str:
    """Consistent color for a MODEL_REGISTRY config: hue = architecture, shade = pretrain source."""
    entry = get_model_entry(model_key)
    base = ARCH_HUE[entry.architecture]
    key = _pretrain_key(entry)
    if key == "supervised":
        return shade(base, 0.45)
    if key == "gastronet_ssl":
        return shade(base, -0.30)
    return base


def config_label(model_key: str, *, short: bool = True) -> str:
    """Display name for one of the 9 MODEL_REGISTRY configs.

    short=True  -> "{ARCH_SHORT} {CORPUS_SHORT} {METHOD_SHORT}", e.g.
                    "RN50 GN-5M DINOv1" (<=21 chars over all 9 configs).
    short=False -> "{architecture}, {corpus} / {method}[ ({variant})]", e.g.
                    "ResNet-50, GastroNet-5M / Self-sup. (DINOv1)".

    Never substitute the result into a DataFrame index, a dict key, an
    MLflow tag filter, a join key, or a filename -- like config_color, this
    raises ValueError (via get_model_entry) on any non-registry string.
    """
    entry = get_model_entry(model_key)
    corpus = _corpus(entry)
    if short:
        arch = ARCH_SHORT[entry.architecture]
        method = METHOD_SHORT.get(entry.pretrain_method, entry.pretrain_method)
        return f"{arch} {CORPUS_SHORT.get(corpus, corpus)} {method}"
    variant = f" ({entry.pretrain_variant})" if entry.pretrain_variant else ""
    return f"{entry.architecture}, {corpus} / {entry.pretrain_method}{variant}"


def config_labels(model_keys: Iterable[str], *, short: bool = True) -> list[str]:
    """Vectorised config_label."""
    return [config_label(k, short=short) for k in model_keys]


def metric_label(metric: str) -> str:
    """Display name for a lowercase metric key. Unknown metrics pass through unchanged."""
    return METRIC_LABEL.get(metric, metric)


def level_label(level: str) -> str:
    """Display name for 'frame' / 'clip'. Unknown levels pass through unchanged."""
    return LEVEL_LABEL.get(level, level)


def slug(text: str) -> str:
    """Filename-safe fragment, e.g. 'ViT-Base/16' -> 'vitbase16'."""
    return text.lower().replace("-", "").replace("/", "").replace(" ", "_").replace(".", "")
