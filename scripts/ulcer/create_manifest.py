"""Stage 4 — Create ulcer train/val/test manifest from processed (or filtrated) frames.

Scans input_dir, applies ulcer-size lookup from the annotations Excel file,
and produces patient-level stratified splits.

Input : data/ulcer/filtrated/{Ulcer,NonUlcer}/vid*/segment_*/*.jpg
        data/ulcer/raw/Ulcer and Non-Ulcer Timestamps.xlsx
Output: data/ulcer/splits/{dataset_manifest.csv, split_info.json,
                            train.csv, val.csv, test.csv}

Usage
-----
    python scripts/ulcer/create_manifest.py
    python scripts/ulcer/create_manifest.py --input-dir data/ulcer/processed  # skip filter stage
    python scripts/ulcer/create_manifest.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.config.paths import get_default_paths
from src.config.preprocessing import SplitConfigBase
from src.data.constants import SIZE_MAP
from src.data.splits import STRAT_MODES, assign_train_val_test_split, build_strat_bin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig(SplitConfigBase):
    input_dir: str = "data/ulcer/filtrated"
    splits_dir: str = "data/ulcer/splits"
    timestamps_file: str = "data/ulcer/raw/Ulcer and Non-Ulcer Timestamps.xlsx"
    timestamps_sheet: str = "Ulcer timestamps"
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")
    class_names: dict = field(default_factory=lambda: {"Ulcer": 1, "NonUlcer": 0})
    strat_mode: str = "ulcer_ratio"


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    image_path: str
    video_id: str
    patient_id: str
    class_name: str
    label: int
    segment_id: str
    segment_number: int
    frame_number: int
    clip_key: str
    relative_path: str
    ulcer_size: int | None


# ---------------------------------------------------------------------------
# Ulcer size helpers
# ---------------------------------------------------------------------------


def load_ulcer_size_lookup(
    timestamps_file: str, sheet_name: str
) -> dict[tuple[str, int], int | None]:
    path = Path(timestamps_file)
    if not path.exists():
        logger.warning("Timestamps file not found: %s. ulcer_size will be NaN.", path)
        return {}
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception as exc:
        logger.warning("Could not read sheet '%s' from %s: %s.", sheet_name, path, exc)
        return {}
    required_cols = {"record_id", "sample_number", "Size:"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Missing columns %s in '%s'.", missing, sheet_name)
        return {}
    lookup: dict[tuple[str, int], int | None] = {}
    for _, row in df.iterrows():
        record_id = str(row["record_id"]).strip()
        try:
            sample_number = int(row["sample_number"])
        except (ValueError, TypeError):
            continue
        raw_size = str(row["Size:"]).strip() if pd.notna(row["Size:"]) else ""
        lookup[(record_id, sample_number)] = SIZE_MAP.get(raw_size, None)
    logger.info("Loaded %d ulcer-size entries from '%s'.", len(lookup), sheet_name)
    return lookup


def _parse_segment_number(segment_id: str) -> int:
    try:
        return int(segment_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def _extract_frame_number(filename: str) -> int:
    try:
        parts = filename.replace(".jpg", "").replace(".png", "").split("_")
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


# ---------------------------------------------------------------------------
# Dataset scanner
# ---------------------------------------------------------------------------


class DatasetPreparer:
    def __init__(self, config: DatasetConfig | None = None):
        self.config = config or DatasetConfig()
        self.records: list[ImageRecord] = []
        self.patient_info: dict = {}

    def scan_directory(self, input_dir: Path, size_lookup: dict) -> pd.DataFrame:
        logger.info("Scanning directory: %s", input_dir)
        for class_name, label in self.config.class_names.items():
            class_dir = input_dir / class_name
            if not class_dir.exists():
                logger.warning("Class directory not found: %s", class_dir)
                continue
            for video_dir in sorted(class_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                video_id = video_dir.name
                if video_id not in self.patient_info:
                    self.patient_info[video_id] = {
                        "video_id": video_id,
                        "has_ulcer": False,
                        "has_non_ulcer": False,
                        "ulcer_frames": 0,
                        "non_ulcer_frames": 0,
                        "total_frames": 0,
                        "segments": [],
                    }
                for segment_dir in sorted(video_dir.iterdir()):
                    if not segment_dir.is_dir():
                        continue
                    segment_id = segment_dir.name
                    segment_number = _parse_segment_number(segment_id)
                    self.patient_info[video_id]["segments"].append(segment_id)
                    ulcer_size = size_lookup.get((video_id, segment_number)) if label == 1 else None
                    for img_path in sorted(segment_dir.iterdir()):
                        if img_path.suffix.lower() not in self.config.image_extensions:
                            continue
                        record = ImageRecord(
                            image_path=str(img_path.absolute()),
                            video_id=video_id,
                            patient_id=video_id,
                            class_name=class_name,
                            label=label,
                            segment_id=segment_id,
                            segment_number=segment_number,
                            frame_number=_extract_frame_number(img_path.name),
                            relative_path=str(img_path.relative_to(input_dir)),
                            clip_key=str(video_id + "__" + segment_id),
                            ulcer_size=ulcer_size,
                        )
                        self.records.append(record)
                        self.patient_info[video_id]["total_frames"] += 1
                        if label == 1:
                            self.patient_info[video_id]["has_ulcer"] = True
                            self.patient_info[video_id]["ulcer_frames"] += 1
                        else:
                            self.patient_info[video_id]["has_non_ulcer"] = True
                            self.patient_info[video_id]["non_ulcer_frames"] += 1

        logger.info(
            "Found %d images from %d patients.",
            len(self.records),
            len(self.patient_info),
        )
        for info in self.patient_info.values():
            t = info["total_frames"]
            info["ulcer_presence"] = info["ulcer_frames"] / t if t > 0 else 0.0
        return pd.DataFrame([asdict(r) for r in self.records])

    def create_patient_level_splits(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        logger.info("Creating patient-level splits...")
        mode = self.config.strat_mode
        patients = list(self.patient_info.keys())

        # Pre-compute bins for reporting
        patient_to_bin = {pid: build_strat_bin(pid, df, mode) for pid in df["patient_id"].unique()}
        combined_bins = [patient_to_bin.get(p, "unknown") for p in patients]
        bin_counts = Counter(combined_bins)
        rare_patients = [p for p, b in zip(patients, combined_bins) if bin_counts[b] < 3]
        if rare_patients:
            logger.warning(
                "%d patient(s) in rare strata — assigned manually (train > test > val).",
                len(rare_patients),
            )

        df, _ = assign_train_val_test_split(
            df,
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio,
            self.config.random_seed,
            patient_col="patient_id",
            label_col="label",
            strat_fn=lambda pid, df_: build_strat_bin(pid, df_, mode),
        )

        split_assignment = df.groupby("patient_id")["split"].first().to_dict()
        split_info = {
            "config": asdict(self.config),
            "total_patients": len(patients),
            "total_images": len(df),
            "stratification": {
                "method": mode,
                "strata_counts": dict(bin_counts),
                "rare_strata_patients": rare_patients,
            },
            "splits": {
                split: {
                    "patients": [p for p, s in split_assignment.items() if s == split],
                    "n_patients": sum(1 for s in split_assignment.values() if s == split),
                    "n_images": len(df[df["split"] == split]),
                    "n_ulcer": len(df[(df["split"] == split) & (df["label"] == 1)]),
                    "n_non_ulcer": len(df[(df["split"] == split) & (df["label"] == 0)]),
                }
                for split in ("train", "val", "test")
            },
            "patient_info": self.patient_info,
            "patient_bins_distribution": {
                b: combined_bins.count(b) for b in sorted(set(combined_bins))
            },
        }
        for split_name, info in split_info["splits"].items():
            logger.info(
                "  %s: %d patients, %d images (ulcer: %d, non-ulcer: %d)",
                split_name,
                info["n_patients"],
                info["n_images"],
                info["n_ulcer"],
                info["n_non_ulcer"],
            )
        return df, split_info

    def save_outputs(self, df: pd.DataFrame, split_info: dict) -> None:
        splits_dir = Path(self.config.splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(splits_dir / "dataset_manifest.csv", index=False)
        with open(splits_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2, default=str)
        for split_name in ("train", "val", "test"):
            df[df["split"] == split_name].to_csv(splits_dir / f"{split_name}.csv", index=False)
        with open(splits_dir / "patient_info.json", "w") as f:
            json.dump(self.patient_info, f, indent=2)
        logger.info("Outputs saved to %s", splits_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    paths = get_default_paths()
    parser = argparse.ArgumentParser(description="Create ulcer detection manifest.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(paths.ulcer_filtrated_dir),
        help="Directory of (filtrated) processed frames.",
    )
    parser.add_argument("--splits-dir", type=str, default=str(paths.ulcer_splits_dir))
    parser.add_argument(
        "--timestamps-file",
        type=str,
        default=str(paths.ulcer_raw_dir / "Ulcer and Non-Ulcer Timestamps.xlsx"),
    )
    parser.add_argument("--timestamps-sheet", type=str, default="Ulcer timestamps")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strat-mode",
        choices=STRAT_MODES,
        default="ulcer_ratio",
        help=(
            "Patient stratification strategy: "
            "'ulcer_ratio' = no_ulcer / low_ulcer (<40%%) / high_ulcer (≥40%%) (default), "
            "'presence' = binary ulcer/no_ulcer, "
            "'size' = dominant ulcer size, "
            "'size_and_presence' = presence × size."
        ),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)

    config = DatasetConfig(
        input_dir=str(input_dir),
        splits_dir=args.splits_dir,
        timestamps_file=args.timestamps_file,
        timestamps_sheet=args.timestamps_sheet,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        strat_mode=args.strat_mode,
    )

    size_lookup = load_ulcer_size_lookup(args.timestamps_file, args.timestamps_sheet)
    preparer = DatasetPreparer(config)
    df = preparer.scan_directory(input_dir, size_lookup)

    if df.empty:
        raise RuntimeError(f"No images found in: {input_dir}")

    df, split_info = preparer.create_patient_level_splits(df)
    preparer.save_outputs(df, split_info)

    print("=" * 72)
    print("ULCER MANIFEST CREATION DONE")
    print("=" * 72)
    print(f"Input dir      : {input_dir}")
    print(f"Total patients : {split_info['total_patients']}")
    print(f"Total images   : {split_info['total_images']}")
    for split_name, info in split_info["splits"].items():
        print(f"  {split_name}: {info['n_patients']} patients, {info['n_images']} images")
    print(f"Splits dir     : {args.splits_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main(build_parser().parse_args())
