"""Shared preprocessing configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SplitConfigBase:
    """Common split configuration used by preprocessing scripts."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, "Split ratios must sum to 1.0"
