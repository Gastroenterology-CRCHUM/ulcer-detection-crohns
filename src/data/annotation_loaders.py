"""Annotation Excel loaders for each colonoscopy pipeline."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _hms_to_seconds(hms) -> float | None:
    """Convert HH:MM:SS string or timedelta to total seconds."""
    if pd.isna(hms):
        return None
    if hasattr(hms, "total_seconds"):
        return float(hms.total_seconds())
    try:
        parts = str(hms).strip().split(":")
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return None


def load_ulcer_annotations(excel_path: Path) -> pd.DataFrame:
    """Load ulcer/non-ulcer timestamps from Excel → unified DataFrame.

    Expected sheets: "ulcer", "non_ulcer".
    Expected columns: record_id, start_time (HH:MM:SS), end_time, sample_number.

    Returns
    -------
    DataFrame with columns: record_id, start_s, end_s, sample_number, label
        label: 1 = ulcer, 0 = non-ulcer
    """
    dfs = []
    for sheet, label in (("ulcer", 1), ("non_ulcer", 0)):
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet)
        except Exception as exc:
            raise ValueError(f"Cannot read sheet '{sheet}' from {excel_path}: {exc}") from exc
        df = df.rename(columns={"start_time": "start_hms", "end_time": "end_hms"})
        df["label"] = label
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["start_s"] = out["start_hms"].apply(_hms_to_seconds)
    out["end_s"] = out["end_hms"].apply(_hms_to_seconds)
    out = out.dropna(subset=["record_id", "start_s", "end_s"]).copy()
    out = out[out["end_s"] > out["start_s"]].copy()
    out["record_id"] = out["record_id"].astype(str).str.strip()
    out = out.sort_values(["record_id", "start_s"]).reset_index(drop=True)
    present = [
        c for c in ["record_id", "start_s", "end_s", "sample_number", "label"] if c in out.columns
    ]
    return out[present]
