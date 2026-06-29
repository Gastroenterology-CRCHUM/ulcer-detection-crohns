"""
scripts/noninformative/preprocess_inf.py
=========================================
Dataset preparation for the Informative / Non-Informative classification task.

Reads 1920 × 1080 raw frames from raw_inf/, applies an octagonal ROI crop
that removes the black scope border and reduces the image to 1350 × 1080,
then creates sample-level train/val/test splits.

Input structure
---------------
    data/raw_inf/
    ├── Informative/
    │   └── {record_id}__sample_{N}__frame_{XXXXXX}.jpg
    └── Non-Informative/
        ├── Blur/
        └── ...

Output structure
----------------
    data/processed_inf/    ← same sub-tree, cropped images
    data/splits_inf/
    ├── dataset_manifest.csv
    ├── split_info.json
    ├── train.csv  /  val.csv  /  test.csv

Manifest columns
----------------
    image_path, patient_id, class_name, label,
    sample_id, unique_sample_id, frame_number,
    cause, relative_path, split

Label encoding
--------------
    1 → Informative
    0 → Non-Informative

Usage
-----
    python -m scripts.noninformative.preprocess_inf

    # Custom paths
    python -m scripts.noninformative.preprocess_inf --raw-inf-dir data/raw_inf --output-dir data/processed_inf --splits-dir data/splits_inf

    # Skip preprocessing (if already done) and regenerate splits only
    python -m scripts.noninformative.preprocess_inf --skip-preprocess

    # After adding raw data, doesn't preprocess the previous ones
    python -m scripts.noninformative.preprocess_inf --incremental
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from scripts.data.preprocess_frames import preprocess_frames
from src.config.paths import get_default_paths
from src.config.preprocessing import SplitConfigBase
from src.data.splits import split_with_rare_strata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InfDatasetConfig(SplitConfigBase):
    raw_inf_dir: str = "data/raw_inf"
    output_dir: str = "data/processed_inf"
    splits_dir: str = "data/splits_inf"
    image_extensions: tuple = (".jpg", ".jpeg", ".png")
    n_jobs: int = -1
    jpeg_quality: int = 95

    class_names: dict = field(
        default_factory=lambda: {
            "Informative": 1,
            "Non-Informative": 0,
        }
    )


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_PATTERN = re.compile(
    r"^(?P<record_id>.+?)__(?P<source>sample|ulcer_NonUlcer|ulcer_Ulcer)_(?P<sample_n>\d+)__frame_(?P<frame_n>\d+)",
    re.IGNORECASE,
)


def parse_filename(stem: str):
    if m := _FNAME_PATTERN.match(stem):
        unique_sample_id = f"{m.group('source')}_{m.group('sample_n')}"
        return m.group("record_id"), unique_sample_id, int(m.group("frame_n"))
    return None


# ---------------------------------------------------------------------------
# Directory scanner → manifest
# ---------------------------------------------------------------------------


def scan_directory(data_root: Path, config: InfDatasetConfig) -> tuple[pd.DataFrame, dict]:
    """Walk processed_inf/ and build the manifest DataFrame."""
    records = []
    patient_info = {}

    for class_name, label in config.class_names.items():
        class_dir = data_root / class_name
        if not class_dir.exists():
            logger.warning("Class directory not found: %s", class_dir)
            continue

        logger.info("Scanning %s …", class_dir)

        if label == 1:  # Informative — flat directory
            img_dirs = [(class_dir, "")]
        else:  # Non-Informative — cause sub-folders
            img_dirs = [
                (item, item.name.replace("_", " "))
                for item in sorted(class_dir.iterdir())
                if item.is_dir()
            ]
            if any(
                f.suffix.lower() in config.image_extensions
                for f in class_dir.iterdir()
                if f.is_file()
            ):
                img_dirs.append((class_dir, ""))

        for img_dir, cause in img_dirs:
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in config.image_extensions:
                    continue

                parsed = parse_filename(img_path.stem)
                if parsed is None:
                    logger.warning("Cannot parse filename: %s — skipped.", img_path.name)
                    continue

                record_id, unique_sample_id, frame_n = parsed
                sample_id = f"{record_id}__{unique_sample_id}"

                if record_id not in patient_info:
                    patient_info[record_id] = {
                        "patient_id": record_id,
                        "has_informative": False,
                        "has_non_informative": False,
                        "informative_frames": 0,
                        "non_informative_frames": 0,
                        "total_frames": 0,
                        "samples": set(),
                        "causes": set(),
                    }

                pi = patient_info[record_id]
                pi["total_frames"] += 1
                pi["samples"].add(sample_id)

                if label == 1:
                    pi["has_informative"] = True
                    pi["informative_frames"] += 1
                else:
                    pi["has_non_informative"] = True
                    pi["non_informative_frames"] += 1
                    if cause:
                        pi["causes"].add(cause)

                records.append(
                    {
                        "image_path": str(img_path.absolute()),
                        "patient_id": record_id,
                        "class_name": class_name,
                        "label": label,
                        "sample_id": sample_id,
                        "unique_sample_id": unique_sample_id,
                        "frame_number": frame_n,
                        "cause": cause,
                        "relative_path": str(img_path.relative_to(data_root)),
                    }
                )

    for pi in patient_info.values():
        pi["samples"] = sorted(pi["samples"])
        pi["causes"] = sorted(pi["causes"])
        t = pi["total_frames"]
        pi["non_informative_ratio"] = pi["non_informative_frames"] / t if t > 0 else 0.0

    df = pd.DataFrame(records)
    logger.info(
        "Found %d frames | %d samples | %d patients",
        len(df),
        df["sample_id"].nunique() if not df.empty else 0,
        df["patient_id"].nunique() if not df.empty else 0,
    )
    return df, patient_info


# ---------------------------------------------------------------------------
# Sample-level splits
# ---------------------------------------------------------------------------


def _strat_bin_frame(row: pd.Series) -> str:
    """Stratification key per frame: ``{class_bin}__{cause_slug}``"""
    if row["label"] == 1:
        return "informative"
    cause = str(row.get("cause", "")).strip()
    cause_slug = re.sub(r"\s+", "_", cause) if cause else "none"
    return f"non_informative__{cause_slug}"


def create_frame_level_splits(
    df: pd.DataFrame,
    patient_info: dict,
    config: InfDatasetConfig,
) -> tuple[dict, pd.DataFrame]:
    """Frame-level stratified splits (train/val/test) via split_with_rare_strata."""
    logger.info("Creating frame-level splits …")

    df = df.copy()
    df["_strat_bin"] = df.apply(_strat_bin_frame, axis=1)
    strat_labels = df["_strat_bin"].tolist()
    strat_counts = {b: strat_labels.count(b) for b in set(strat_labels)}

    for b, cnt in strat_counts.items():
        if cnt < 3:
            logger.warning("  stratum '%s' (rare, %d frames) — assigned manually.", b, cnt)

    train_idx, val_idx, test_idx, _, _ = split_with_rare_strata(
        df.index.tolist(),
        strat_labels,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed,
    )
    split_map = {i: "train" for i in train_idx}
    split_map.update({i: "val" for i in val_idx})
    split_map.update({i: "test" for i in test_idx})

    df["split"] = df.index.map(split_map)
    df = df.drop(columns=["_strat_bin"])

    split_info = {
        "config": asdict(config),
        "total_frames": len(df),
        "stratification": {
            "method": "label x cause (frame-level)",
            "strata_counts": strat_counts,
        },
        "splits": {
            split: {
                "n_frames": len(df[df["split"] == split]),
                "n_informative": len(df[(df["split"] == split) & (df["label"] == 1)]),
                "n_non_informative": len(df[(df["split"] == split) & (df["label"] == 0)]),
            }
            for split in ("train", "val", "test")
        },
        "patient_info": patient_info,
    }

    for name, info in split_info["splits"].items():
        logger.info(
            "  %s: %d frames (inf: %d, non-inf: %d)",
            name,
            info["n_frames"],
            info["n_informative"],
            info["n_non_informative"],
        )

    return split_info, df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_outputs(df: pd.DataFrame, split_info: dict, patient_info: dict, config: InfDatasetConfig):
    splits_dir = Path(config.splits_dir)
    output_dir = Path(config.output_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(splits_dir / "dataset_manifest.csv", index=False)
    with open(splits_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2, default=str)
    for split in ("train", "val", "test"):
        df[df["split"] == split].to_csv(splits_dir / f"{split}.csv", index=False)
    with open(output_dir / "patient_info.json", "w") as f:
        json.dump(patient_info, f, indent=2, default=str)

    logger.info("Splits → %s", splits_dir)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare(
    config: InfDatasetConfig | None = None,
    skip_preprocess: bool = False,
    incremental: bool = False,
) -> tuple[pd.DataFrame, dict]:
    config = config or InfDatasetConfig()
    raw_root = Path(config.raw_inf_dir)
    out_root = Path(config.output_dir)

    try:
        from src.data.file_utils import check_and_abort

        if not check_and_abort(raw_root, verbose=True):
            raise SystemExit(
                f"Duplicates found in raw_inf/. "
                f"Fix with: python scripts/data/check_duplicates.py --raw-inf {raw_root} --fix"
            )
    except ImportError:
        logger.warning("check_duplicates not found — skipping duplicate check.")

    logger.info("=" * 60)
    logger.info("Informative/Non-Informative dataset preparation")
    logger.info("=" * 60)

    if skip_preprocess:
        if not out_root.exists():
            raise FileNotFoundError(f"{out_root} not found. Run without --skip-preprocess first.")
        logger.info("Skipping preprocessing — using %s", out_root)
    else:
        preprocess_frames(
            raw_root,
            out_root,
            jpeg_quality=config.jpeg_quality,
            incremental=incremental,
            default_platform="olympus",
        )

    df, patient_info = scan_directory(out_root, config)
    if df.empty:
        raise ValueError("No images found. Check raw_inf_dir and directory structure.")

    split_info, df = create_frame_level_splits(df, patient_info, config)
    save_outputs(df, split_info, patient_info, config)

    logger.info("Done.")
    return df, split_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    paths = get_default_paths()
    inf_config = paths.get_informative_config()

    parser = argparse.ArgumentParser(
        description="Prepare Informative/Non-Informative dataset (1920×1080 → 1350×1080)"
    )
    parser.add_argument("--raw-inf-dir", default=str(inf_config["raw_dir"]))
    parser.add_argument("--output-dir", default=str(inf_config["processed_dir"]))
    parser.add_argument("--splits-dir", default=str(inf_config["splits_dir"]))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip image preprocessing and regenerate splits only.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process frames that don't already exist in output-dir.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main(args: argparse.Namespace) -> None:
    if args.incremental and args.skip_preprocess:
        raise ValueError("--incremental and --skip-preprocess are mutually exclusive.")

    config = InfDatasetConfig(
        raw_inf_dir=args.raw_inf_dir,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
        jpeg_quality=args.jpeg_quality,
    )
    df, split_info = prepare(
        config, skip_preprocess=args.skip_preprocess, incremental=args.incremental
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total frames  : {split_info['total_frames']}")
    print("\nSplits:")
    for name, info in split_info["splits"].items():
        print(
            f"  {name:<6}: {info['n_frames']} frames "
            f"(inf: {info['n_informative']}, non-inf: {info['n_non_informative']})"
        )
    if "cause" in df.columns:
        print("\nNon-Informative cause distribution:")
        for cause, cnt in df[df["label"] == 0]["cause"].value_counts(dropna=False).items():
            print(f"  {cause or '(no cause)':<35} {cnt:>6} frames")
    print("=" * 60)


if __name__ == "__main__":
    main(build_parser().parse_args())
