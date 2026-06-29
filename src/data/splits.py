"""Patient-level stratification and split helpers.

Shared between:
  - scripts/ulcer/create_manifest.py  — ulcer train/val/test splits (STRAT_MODES, build_strat_bin)
  - src/data/mes.py                   — MES inference splits
  - src/data/dataloader.py            — CV folds and val carve-outs at training time
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def modal_patient_label(
    patient_id: str,
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> str:
    """Most frequent label for *patient_id* as a string, or 'unknown'."""
    mask = df[patient_col].astype(str) == str(patient_id)
    vals = df.loc[mask, label_col].dropna()
    if vals.empty:
        return "unknown"
    mode_vals = vals.astype(int).mode()
    return str(int(mode_vals.iloc[0])) if not mode_vals.empty else "unknown"


def patient_label_array(
    df: pd.DataFrame,
    strat_fn: Callable[[str, pd.DataFrame], str],
    patient_col: str = "patient_id",
) -> np.ndarray:
    """Return a per-frame array of stratification labels produced by *strat_fn*."""
    label_map = {pid: strat_fn(pid, df) for pid in df[patient_col].unique()}
    return np.asarray(df[patient_col].map(label_map).values)


# ---------------------------------------------------------------------------
# Ulcer-specific stratification
# ---------------------------------------------------------------------------


STRAT_MODES = ("size", "presence", "size_and_presence", "ulcer_ratio")


def dominant_ulcer_size(patient_id: str, df: pd.DataFrame) -> str:
    """Return the most frequent ulcer size category for a patient, or 'none'."""
    rows = df[(df["patient_id"] == patient_id) & (df["label"] == 1) & df["ulcer_size"].notna()]
    if rows.empty:
        return "none"
    mode_val = rows["ulcer_size"].mode()
    return str(int(mode_val.iloc[0])) if not mode_val.empty else "none"


def ulcer_presence_bin(patient_id: str, df: pd.DataFrame) -> str:
    """Binary ulcer presence for a patient: 'no_ulcer' or 'ulcer'."""
    has_ulcer = (df[df["patient_id"] == patient_id]["label"] == 1).any()
    return "ulcer" if has_ulcer else "no_ulcer"


def patient_strat_label(patient_id: str, df: pd.DataFrame) -> str:
    """Coarse ulcer bin: no_ulcer / low_ulcer (<40%) / high_ulcer (≥40%)."""
    ratio = df[df["patient_id"] == patient_id]["label"].mean()
    if ratio == 0:
        return "no_ulcer"
    return "low_ulcer" if ratio < 0.40 else "high_ulcer"


def patient_strat_labels(df: pd.DataFrame) -> np.ndarray:
    """Per-frame ulcer stratification labels array (wraps patient_label_array)."""
    return patient_label_array(df, patient_strat_label)


def build_strat_bin(patient_id: str, df: pd.DataFrame, mode: str) -> str:
    """Return a stratification bin string for the given mode.

    mode='size'              → dominant size only   ('none', '0', '1', '2')
    mode='presence'          → binary presence      ('no_ulcer', 'ulcer')
    mode='size_and_presence' → presence × size      ('ulcer__1', 'no_ulcer__none', …)
    """
    if mode == "ulcer_ratio":
        return patient_strat_label(patient_id, df)
    if mode == "size":
        has_size = "ulcer_size" in df.columns and df["ulcer_size"].notna().any()
        return (
            dominant_ulcer_size(patient_id, df) if has_size else ulcer_presence_bin(patient_id, df)
        )
    if mode == "presence":
        return ulcer_presence_bin(patient_id, df)
    if mode == "size_and_presence":
        has_size = "ulcer_size" in df.columns and df["ulcer_size"].notna().any()
        presence = ulcer_presence_bin(patient_id, df)
        size = dominant_ulcer_size(patient_id, df) if has_size else "none"
        return f"{presence}__{size}"
    raise ValueError(f"Unknown strat_mode: {mode!r}. Choose from {STRAT_MODES}.")


# ---------------------------------------------------------------------------
# Core split primitive with rare-strata handling
# ---------------------------------------------------------------------------


def split_with_rare_strata(
    ids: list,
    strat_labels: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    rare_threshold: int = 3,
) -> tuple[list, list, list, str, list]:
    """Stratified 3-way split with manual proportional assignment for rare strata.

    A stratum is treated as rare (manual assignment) when it has fewer members
    than the auto-calibrated threshold: max(rare_threshold, ceil(1/test_ratio)).
    This guarantees that every stratum with too few samples for sklearn to place
    at least one member in test is handled manually instead.

    Manual priority: test first (must be evaluable), then train, val gets the rest.
    Common strata use sklearn stratified split with random fallback.

    Returns (train_ids, val_ids, test_ids, strategy, rare_ids).
    strategy is 'stratified', 'partial' (common stratified, rare manual), or 'random'.
    """
    effective_threshold = max(rare_threshold, math.ceil(1.0 / test_ratio) + 1)
    bin_counts = Counter(strat_labels)
    common_ids, common_labels, rare_ids, rare_labels = [], [], [], []
    for id_, label in zip(ids, strat_labels):
        if bin_counts[label] >= effective_threshold:
            common_ids.append(id_)
            common_labels.append(label)
        else:
            rare_ids.append(id_)
            rare_labels.append(label)

    train_out: list = []
    val_out: list = []
    test_out: list = []

    if rare_ids:
        rng = np.random.default_rng(random_seed)
        rare_by_label: dict[str, list] = {}
        for id_, label in zip(rare_ids, rare_labels):
            rare_by_label.setdefault(label, []).append(id_)
        for label in sorted(rare_by_label):
            group = rng.permutation(rare_by_label[label]).tolist()
            n = len(group)
            if n >= 3:
                # Guarantee at least 1 in each split; train gets the remainder.
                n_test = max(1, round(n * test_ratio))
                n_val = max(1, round(n * val_ratio))
                n_train = n - n_test - n_val
                if n_train < 1:
                    # Reduce val to free one slot for train.
                    n_val = max(0, n_val - (1 - n_train))
                    n_train = n - n_test - n_val
            elif n == 2:
                n_test, n_val, n_train = 1, 0, 1
            else:
                n_test, n_val, n_train = 1, 0, 0
            test_out += group[:n_test]
            train_out += group[n_test : n_test + n_train]
            val_out += group[n_test + n_train :]

    strategy = "partial" if rare_ids else "stratified"
    if common_ids:
        try:
            tr_ids, vt_ids, _, vt_labels = train_test_split(
                common_ids,
                common_labels,
                test_size=(val_ratio + test_ratio),
                random_state=random_seed,
                stratify=common_labels,
            )
            relative_test = test_ratio / (val_ratio + test_ratio)
            try:
                v_ids, t_ids = train_test_split(
                    vt_ids,
                    test_size=relative_test,
                    random_state=random_seed,
                    stratify=vt_labels,
                )
            except ValueError:
                v_ids, t_ids = train_test_split(
                    vt_ids, test_size=relative_test, random_state=random_seed
                )
        except ValueError:
            strategy = "random"
            tr_ids, vt_ids = train_test_split(
                common_ids, test_size=(val_ratio + test_ratio), random_state=random_seed
            )
            relative_test = test_ratio / (val_ratio + test_ratio)
            v_ids, t_ids = train_test_split(
                vt_ids, test_size=relative_test, random_state=random_seed
            )
        train_out += list(tr_ids)
        val_out += list(v_ids)
        test_out += list(t_ids)

    return train_out, val_out, test_out, strategy, rare_ids


# ---------------------------------------------------------------------------
# CV fold assignment
# ---------------------------------------------------------------------------


def assign_cv_folds(
    train_df: pd.DataFrame,
    n_splits: int,
    random_seed: int,
    strat_fn: Callable[[str, pd.DataFrame], str] | None = None,
) -> pd.DataFrame:
    """Add a `fold` column (0 … n_splits-1) to *train_df*.

    StratifiedGroupKFold guarantees:
      - All frames of a patient stay in the same fold (no leakage).
      - Folds share similar label distributions.

    *strat_fn* maps (patient_id, df) → bin string.
    Defaults to modal patient label (works for binary and multiclass).

    Returns a copy of train_df with the new `fold` column.
    """
    if strat_fn is None:
        strat_fn = modal_patient_label
    strat_labels = patient_label_array(train_df, strat_fn)
    groups = np.asarray(train_df["patient_id"].astype(str).values)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_col = np.empty(len(train_df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(X=train_df, y=strat_labels, groups=groups)):
        fold_col[val_idx] = fold_idx

    train_df = train_df.copy()
    train_df["fold"] = fold_col
    return train_df


# ---------------------------------------------------------------------------
# Single val-split carve-out
# ---------------------------------------------------------------------------


def assign_val_split(
    train_df: pd.DataFrame,
    val_ratio: float,
    random_seed: int,
    strat_fn: Callable[[str, pd.DataFrame], str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a patient-level val set from *train_df*.

    Falls back to unstratified if any stratum has fewer than 2 patients.
    *strat_fn* defaults to modal patient label.

    Returns (new_train_df, val_df) with no patient overlap.
    """
    if strat_fn is None:
        strat_fn = modal_patient_label
    patients = train_df["patient_id"].unique().tolist()
    label_map = {pid: strat_fn(pid, train_df) for pid in patients}
    strat_bins = [label_map[p] for p in patients]

    try:
        train_patients, val_patients = train_test_split(
            patients,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=strat_bins,
        )
    except ValueError:
        train_patients, val_patients = train_test_split(
            patients,
            test_size=val_ratio,
            random_state=random_seed,
        )

    return (
        train_df[train_df["patient_id"].isin(train_patients)].copy(),
        train_df[train_df["patient_id"].isin(val_patients)].copy(),
    )


# ---------------------------------------------------------------------------
# Full train / val / test split
# ---------------------------------------------------------------------------


def assign_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    patient_col: str = "patient_id",
    label_col: str = "label",
    strat_fn: Callable[[str, pd.DataFrame], str] | None = None,
    rare_threshold: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Split a frame-level dataset by patient, stratified on the modal patient label.

    Keeps all frames of a patient in the same split. Works for binary and
    multiclass labels (ulcer 0/1, MES Mayo 0-3). Rare strata (count <
    rare_threshold) are assigned manually; otherwise stratified split is used.

    *strat_fn* maps (patient_id, df) → bin string; overrides modal label when
    richer stratification is needed (e.g. ulcer size × presence).

    Returns (df_with_split_column, split_info_dict).
    """
    if patient_col not in df.columns:
        raise KeyError(f"Missing required column: {patient_col}")
    if label_col not in df.columns and strat_fn is None:
        raise KeyError(f"Missing required column: {label_col}")

    patients = sorted(df[patient_col].dropna().astype(str).unique().tolist())
    if not patients:
        out = df.copy()
        out["split"] = pd.Series(dtype=str)
        return out, {
            "strategy": "empty",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
            "splits": {
                split: {"n_patients": 0, "n_frames": 0, "label_counts": {}}
                for split in ("train", "val", "test")
            },
        }

    label_map = {
        pid: (strat_fn if strat_fn is not None else patient_strat_label)(pid, df)
        for pid in patients
    }
    strat_labels = [label_map[pid] for pid in patients]

    train_ids, val_ids, test_ids, strategy, _ = split_with_rare_strata(
        patients, strat_labels, train_ratio, val_ratio, test_ratio, random_seed, rare_threshold
    )

    split_map: dict[str, str] = {str(pid): "train" for pid in train_ids}
    split_map.update({str(pid): "val" for pid in val_ids})
    split_map.update({str(pid): "test" for pid in test_ids})

    out = df.copy()
    out[patient_col] = out[patient_col].astype(str)
    out["split"] = out[patient_col].map(split_map)

    split_info = {
        "strategy": strategy,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "splits": {
            split_name: {
                "n_patients": int(out.loc[out["split"] == split_name, patient_col].nunique()),
                "n_frames": int((out["split"] == split_name).sum()),
                "label_counts": (
                    {
                        int(str(lbl)): int(cnt)
                        for lbl, cnt in cast(
                            pd.Series, out.loc[out["split"] == split_name, label_col]
                        )
                        .value_counts()
                        .sort_index()
                        .items()
                    }
                    if label_col in out.columns
                    else {}
                ),
            }
            for split_name in ("train", "val", "test")
        },
    }
    return out, split_info
