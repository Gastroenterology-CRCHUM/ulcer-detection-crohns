"""Build the temporal held-out manifest from data/ulcer/heldout/{Ulcer,NonUlcer}.

Every row gets split="heldout" — this is not a train/val/test split of the
main cohort (see scripts/ulcer/create_manifest.py for that). It's an
independent patient cohort, evaluated post-hoc against already-trained
models. See data/ulcer/heldout/README.md.

Usage
-----
    python -m scripts.ulcer.create_heldout_manifest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.paths import get_default_paths

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {"Ulcer": 1, "NonUlcer": 0}


def build_heldout_manifest(input_dir: Path) -> pd.DataFrame:
    """Scan input_dir/{Ulcer,NonUlcer}/<video_id>/<segment>/*.jpg into a manifest."""
    rows: list[dict] = []
    for class_name, label in CLASS_NAMES.items():
        class_dir = input_dir / class_name
        if not class_dir.exists():
            continue
        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            for segment_dir in sorted(video_dir.iterdir()):
                if not segment_dir.is_dir():
                    continue
                segment_id = segment_dir.name
                for img_path in sorted(segment_dir.iterdir()):
                    if img_path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    rows.append(
                        {
                            "relative_path": str(img_path.relative_to(input_dir)),
                            "video_id": video_id,
                            "patient_id": video_id,
                            "clip_key": f"{video_id}__{segment_id}",
                            "label": label,
                            "split": "heldout",
                        }
                    )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    paths = get_default_paths()
    parser = argparse.ArgumentParser(
        description="Build the temporal held-out manifest from data/ulcer/heldout/{Ulcer,NonUlcer}."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(paths.ulcer_heldout_dir),
        help="Directory containing Ulcer/ and NonUlcer/ (default: data/ulcer/heldout).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(paths.ulcer_heldout_dir / "heldout_temporal_manifest.csv"),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    df = build_heldout_manifest(input_dir)
    if df.empty:
        raise RuntimeError(f"No images found under {input_dir}/Ulcer or {input_dir}/NonUlcer.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_ulcer = int((df["label"] == 1).sum())
    n_non_ulcer = int((df["label"] == 0).sum())
    print("=" * 60)
    print("HELDOUT MANIFEST DONE")
    print("=" * 60)
    print(f"Patients : {df['patient_id'].nunique()}")
    print(f"Clips    : {df['clip_key'].nunique()}")
    print(f"Frames   : {len(df)}  (ulcer: {n_ulcer}, non-ulcer: {n_non_ulcer})")
    print(f"Output   : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main(build_parser().parse_args())
