"""Probability helpers for model outputs.

Canonical location for converting logits to probabilities.
"""

import torch


def logits_to_probs(logits, num_classes):
    """Convert model logits to probabilities for binary or multiclass tasks."""
    return torch.sigmoid(logits.squeeze(1)) if num_classes == 1 else torch.softmax(logits, dim=1)
