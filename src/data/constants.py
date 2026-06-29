"""
Shared constants for ulcer size encoding.
"""

# Encoding used in dataset_manifest.csv (column `ulcer_size`)
SIZE_MAP: dict[str, int] = {
    "<5mm": 1,
    "5-20mm": 2,
    ">20mm": 3,
}

SIZE_LABELS: dict[int, str] = {v: k for k, v in SIZE_MAP.items()}

# Canonical display order (includes the "unknown" sentinel)
SIZE_ORDER: list[str] = ["<5mm", "5-20mm", ">20mm", "unknown"]

# Default number of cross-validation folds
N_FOLDS: int = 5
