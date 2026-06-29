"""
DataLoader factory supporting two training strategies:

    "cv"    → k-fold patient-level cross-validation.
              For fold i: patients in fold i → val loader,
                          all other train patients → train loader.
              Set use_full_trainset=True to merge manifest val into train
              (recommended for CV — maximises available data).

    "split" → Classic single train/val split.
              Uses manifest 'val' rows if present, otherwise carves
              a val set from 'train' on the fly using val_ratio.

The held-out test set is always accessed via get_test_loader() and is
never touched during training.

Public API
----------
    get_loaders(mode, ...)        → (train_loader, val_loader)
    get_cv_loaders(fold, ...)     → (train_loader, val_loader) for one fold
    get_split_loaders(...)        → (train_loader, val_loader) for single split
    get_test_loader(...)          → test_loader (filters split == "test")
    get_heldout_loader(...)       → held-out test loader (all rows, no split filter)
    get_all_folds(...)            → list[(train_loader, val_loader)]
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.constants import N_FOLDS
from src.data.dataset import UlcerDataset
from src.data.splits import assign_cv_folds, assign_val_split
from src.data.transforms import get_transforms

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _make_loader(
    df: pd.DataFrame,
    data_dir: Path,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    label_col: str = "label",
) -> DataLoader:
    dataset = UlcerDataset(df, data_dir, transform=transform, label_col=label_col)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def _sampling_train(
    train_df: pd.DataFrame,
    subset_ratio: float,
    label_col: str = "label",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified clip-level sampling that preserves:
      1. Class ratio (modal label per clip) from the full train set.
      2. Frame-count distribution per class (tertile bins: few / medium / many).

    Works on the frame-level manifest — groups by clip_key, samples clips,
    then returns all frames belonging to sampled clips.
    Supports binary and multiclass labels.
    """
    # ── 1. Build clip-level summary ────────────────────────────────────────
    clip_df = train_df.groupby("clip_key", as_index=False).agg(
        _label_val=(label_col, lambda x: int(x.mode().iloc[0])),
        n_frames=(label_col, "count"),
        patient_id=("patient_id", "first"),
    )

    # Rename the temporary label column to the actual label_col name
    clip_df = clip_df.rename(columns={"_label_val": label_col})

    n_clips_total = len(clip_df)
    assert n_clips_total > 0, "clip_df is empty — check clip_key construction."

    # ── 2. Frame-count tertile bins (computed on full train set) ───────────
    clip_df["frame_bin"] = pd.Categorical(
        [""] * len(clip_df), categories=["few", "medium", "many"], ordered=True
    )
    for cls_val in clip_df[label_col].unique():
        mask = clip_df[label_col] == cls_val
        cls_frames = clip_df.loc[mask, "n_frames"]
        q33, q66 = cls_frames.quantile([1 / 3, 2 / 3])
        if q33 == q66:
            q33 = q66 - 1
        clip_df.loc[mask, "frame_bin"] = pd.cut(
            cls_frames,
            bins=[-np.inf, q33, q66, np.inf],
            labels=["few", "medium", "many"],
        )

    # ── 3. Stratum = class × frame_bin  ────────────────────────────────────
    clip_df["_stratum"] = clip_df[label_col].astype(str) + "_" + clip_df["frame_bin"].astype(str)

    # Sanity check: no clip should have a NaN stratum
    n_nan_strata = clip_df["_stratum"].isna().sum()
    if n_nan_strata > 0:
        raise ValueError(
            f"_sampling_train: {n_nan_strata} clips have NaN stratum. "
            "Check n_frames column for NaN or zero values."
        )

    # ── 4. Stratified sampling — proportional allocation ───────────────────
    # Compute global target first, then distribute across strata.
    # Per-stratum max(1, ...) inflates small ratios when n_strata > n_target.
    rng = np.random.default_rng(random_seed)
    n_target = max(1, round(n_clips_total * subset_ratio))

    strata_sizes = clip_df.groupby("_stratum", observed=True).size()
    raw_alloc = strata_sizes / strata_sizes.sum() * n_target
    floor_alloc = raw_alloc.apply(np.floor).astype(int)
    remainders = raw_alloc - floor_alloc
    n_remaining = n_target - int(floor_alloc.sum())
    if n_remaining > 0:
        top_strata = remainders.nlargest(n_remaining).index
        floor_alloc[top_strata] += 1

    sampled_parts = []
    for stratum_name, g in clip_df.groupby("_stratum", observed=True):
        quota = int(floor_alloc.get(stratum_name, 0))
        if quota > 0:
            sampled_parts.append(
                g.sample(n=min(quota, len(g)), random_state=int(rng.integers(0, 2**31)))
            )
    sampled_clips = (
        pd.concat(sampled_parts).reset_index(drop=True) if sampled_parts else clip_df.iloc[:0]
    )

    if subset_ratio == 1.0 and len(sampled_clips) != n_clips_total:
        raise ValueError(
            f"_sampling_train: expected {n_clips_total} clips at ratio=1.0, "
            f"got {len(sampled_clips)}. Investigate stratum assignments."
        )

    # ── 5. Filter frame-level manifest ─────────────────────────────────────
    sampled_df = train_df[train_df["clip_key"].isin(sampled_clips["clip_key"])].copy()

    # ── 6. Diagnostics ─────────────────────────────────────────────────────
    n_frames_total = len(train_df)
    n_frames_subset = len(sampled_df)
    n_clips_subset = len(sampled_clips)

    class_counts = sampled_clips[label_col].value_counts().sort_index()
    class_ratio_full = clip_df[label_col].value_counts(normalize=True).sort_index()
    class_ratio_subset = sampled_clips[label_col].value_counts(normalize=True).sort_index()

    all_classes = sorted(class_counts.index)
    col_w = 8
    header = (
        f"  {'label':<10}"
        + f"{'clips':>{col_w}}"
        + f"{'full %':>{col_w}}"
        + f"{'subset %':>{col_w + 2}}"
    )
    print(
        f"\n[subset {subset_ratio:.0%}]  "
        f"clips: {n_clips_subset}/{n_clips_total}  "
        f"frames: {n_frames_subset}/{n_frames_total} "
        f"({n_frames_subset / n_frames_total:.1%})"
    )
    print(header)
    print("  " + "-" * (10 + col_w * 2 + col_w + 2))
    for cls in all_classes:
        n = class_counts.get(cls, 0)
        rf = class_ratio_full.get(cls, 0) * 100
        rs = class_ratio_subset.get(cls, 0) * 100
        print(f"  {cls!s:<10}{n:>{col_w}}{rf:>{col_w}.1f}%{rs:>{col_w + 1}.1f}%")

    bin_summary = (
        sampled_clips.groupby([label_col, "frame_bin"], observed=True).size().unstack(fill_value=0)
    )
    print(f"  frame bins:\n{bin_summary.to_string()}\n")

    return sampled_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_loaders(
    mode: Literal["cv", "split"],
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    subset_ratio: float = 1.0,
    label_col: str = "label",
    *,
    # CV-specific
    fold: int = 0,
    n_splits: int = N_FOLDS,
    use_full_trainset: bool = False,
    use_all_splits: bool = False,
    # Split-specific
    val_ratio: float = 0.15,
    # Shared
    num_workers: int = 8,
    equalize: bool = True,
    random_seed: int = 42,
    **augmentation_params,
) -> tuple[DataLoader, DataLoader]:
    """
    Unified dispatch for both training strategies.

    Args:
        mode:              "cv" or "split".
        fold:              (CV only) fold index in [0, n_splits).
        n_splits:          (CV only) total number of folds.
        use_all_splits:    (CV only) use full manifest regardless of split column.
        use_full_trainset: (CV only) merge manifest 'val' into train pool.
        val_ratio:         (split only) val fraction when no 'val' in manifest.

    Returns:
        (train_loader, val_loader)
    """
    if mode == "cv":
        return get_cv_loaders(
            fold=fold,
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            subset_ratio=subset_ratio,
            num_workers=num_workers,
            equalize=equalize,
            n_splits=n_splits,
            use_full_trainset=use_full_trainset,
            use_all_splits=use_all_splits,
            random_seed=random_seed,
            label_col=label_col,
            **augmentation_params,
        )
    elif mode == "split":
        return get_split_loaders(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            subset_ratio=subset_ratio,
            val_ratio=val_ratio,
            num_workers=num_workers,
            equalize=equalize,
            random_seed=random_seed,
            label_col=label_col,
            **augmentation_params,
        )
    else:
        raise ValueError(f"mode must be 'cv' or 'split', got '{mode}'.")


def get_cv_loaders(
    fold: int,
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    subset_ratio: float = 1.0,
    label_col: str = "label",
    *,
    num_workers: int = 8,
    equalize: bool = True,
    n_splits: int = N_FOLDS,
    use_full_trainset: bool = False,
    use_all_splits: bool = False,
    random_seed: int = 42,
    **augmentation_params,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader) for one cross-validation fold.

    Args:
        fold:              Fold index in [0, n_splits).
        n_splits:          Total number of CV folds.
        use_all_splits:    If True, use the entire manifest (all rows regardless
                           of 'split' column). Recommended for full CV without a
                           held-out test set.
        use_full_trainset: If True and use_all_splits=False, merges manifest
                           'val' rows into the train pool before folding.
        subset_ratio:      If <1.0, randomly subsample the train pool before
                           folding (for faster experiments).

    Returns:
        (train_loader, val_loader)
    """
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits}), got {fold}.")

    manifest = pd.read_csv(manifest_path)

    if use_all_splits:
        train_df = manifest.copy()
    else:
        splits_to_include = ["train", "val"] if use_full_trainset else ["train"]
        train_df = manifest[manifest["split"].isin(splits_to_include)].copy()

    if subset_ratio < 1.0:
        train_df = _sampling_train(train_df, subset_ratio, label_col, random_seed)

    train_df = assign_cv_folds(train_df, n_splits=n_splits, random_seed=random_seed)

    fold_train = train_df[train_df["fold"] != fold]
    fold_val = train_df[train_df["fold"] == fold]

    train_transform = get_transforms(
        img_size, is_training=True, equalize=equalize, **augmentation_params
    )
    val_transform = get_transforms(img_size, is_training=False, equalize=equalize)

    return (
        _make_loader(
            fold_train,
            data_dir,
            train_transform,
            batch_size,
            num_workers,
            label_col=label_col,
            shuffle=True,
        ),
        _make_loader(
            fold_val,
            data_dir,
            val_transform,
            batch_size,
            num_workers,
            label_col=label_col,
            shuffle=False,
        ),
    )


def get_split_loaders(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    subset_ratio: float = 1.0,
    label_col: str = "label",
    *,
    val_ratio: float = 0.15,
    num_workers: int = 8,
    equalize: bool = True,
    random_seed: int = 42,
    **augmentation_params,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader) for a classic single split.

    If the manifest has 'val' rows, they are used directly.
    Otherwise a patient-level val set is carved from 'train' using val_ratio.

    Args:
        val_ratio: Fraction of train patients used for val when no 'val'
                   rows exist in the manifest.
        subset_ratio: If <1.0, randomly subsample the train pool before
                   splitting (for faster experiments).

    Returns:
        (train_loader, val_loader)
    """
    manifest = pd.read_csv(manifest_path)

    if "val" in manifest["split"].values:
        train_df = manifest[manifest["split"] == "train"].copy()
        val_df = manifest[manifest["split"] == "val"].copy()
    else:
        all_train = manifest[manifest["split"] == "train"].copy()
        train_df, val_df = assign_val_split(all_train, val_ratio=val_ratio, random_seed=random_seed)
        print(
            "[!] Validation split not found in manifest — splitting train randomly into train/val."
        )

    if subset_ratio < 1.0:
        train_df = _sampling_train(train_df, subset_ratio, label_col, random_seed)

    train_transform = get_transforms(
        img_size, is_training=True, equalize=equalize, **augmentation_params
    )
    val_transform = get_transforms(img_size, is_training=False, equalize=equalize)

    return (
        _make_loader(
            train_df,
            data_dir,
            train_transform,
            batch_size,
            num_workers,
            label_col=label_col,
            shuffle=True,
        ),
        _make_loader(
            val_df,
            data_dir,
            val_transform,
            batch_size,
            num_workers,
            label_col=label_col,
            shuffle=False,
        ),
    )


def get_test_loader(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    label_col: str = "label",
    *,
    num_workers: int = 8,
    equalize: bool = True,
) -> DataLoader:
    """
    Return the held-out test DataLoader.
    Call only once, after all training and model selection are complete.

    Returns:
        test_loader
    """
    manifest = pd.read_csv(manifest_path)
    test_df = manifest[manifest["split"] == "test"].copy()
    transform = get_transforms(img_size, is_training=False, equalize=equalize)
    return _make_loader(
        test_df, data_dir, transform, batch_size, num_workers, label_col=label_col, shuffle=False
    )


def get_val_loader(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    label_col: str = "label",
    *,
    val_ratio: float = 0.15,
    num_workers: int = 8,
    equalize: bool = True,
    random_seed: int = 42,
) -> DataLoader:
    """Fixed val loader — independent of any subset_ratio."""
    manifest = pd.read_csv(manifest_path)
    if "val" in manifest["split"].values:
        val_df = manifest[manifest["split"] == "val"].copy()
    else:
        all_train = manifest[manifest["split"] == "train"].copy()
        _, val_df = assign_val_split(all_train, val_ratio=val_ratio, random_seed=random_seed)
        print(
            "[!] Validation split not found in manifest — splitting train randomly into train/val."
        )
    transform = get_transforms(img_size, is_training=False, equalize=equalize)
    return _make_loader(
        val_df, data_dir, transform, batch_size, num_workers, label_col=label_col, shuffle=False
    )


def get_heldout_loader(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    label_col: str = "label",
    *,
    num_workers: int = 8,
    equalize: bool = True,
) -> DataLoader:
    """Return a DataLoader for an external held-out test manifest.

    Unlike get_test_loader(), this does not require a split column — all rows
    are included. If a split column is present and contains "test" rows, only
    those are used; otherwise the full manifest is loaded. Use this for a
    manifest that was never part of any train/val/test split assignment.
    """
    manifest = pd.read_csv(manifest_path)
    if "split" in manifest.columns and "test" in manifest["split"].values:
        df = manifest[manifest["split"] == "test"].copy()
    else:
        df = manifest.copy()
    transform = get_transforms(img_size, is_training=False, equalize=equalize)
    return _make_loader(
        df, data_dir, transform, batch_size, num_workers, label_col=label_col, shuffle=False
    )


def get_all_folds(
    manifest_path: Path,
    data_dir: Path,
    batch_size: int,
    img_size: int,
    label_col: str = "label",
    subset_ratio: float = 1.0,
    n_splits: int = N_FOLDS,
    **kwargs,
) -> list[tuple[DataLoader, DataLoader]]:
    """
    Return all (train_loader, val_loader) pairs for every CV fold.

    Example::

        for fold, (train_loader, val_loader) in enumerate(get_all_folds(...)):
            train_one_fold(fold, train_loader, val_loader)

        test_loader = get_test_loader(...)  # only after all folds are done

    Returns:
        List of length n_splits, each element is (train_loader, val_loader).
    """
    return [
        get_cv_loaders(
            fold=k,
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            subset_ratio=subset_ratio,
            n_splits=n_splits,
            label_col=label_col,
            **kwargs,
        )
        for k in range(n_splits)
    ]
