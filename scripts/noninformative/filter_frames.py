"""Generic RF informative-frame filter — shared by ulcer and MES pipelines.

Scans input_dir recursively, classifies frames with the informative RF model,
and copies informative frames to output_dir preserving relative path structure.

Usage
-----
    python scripts/noninformative/filter_frames.py \\
        --input-dir data/ulcer/processed \\
        --output-dir data/ulcer/filtrated

    python scripts/noninformative/filter_frames.py \\
        --input-dir data/mes/processed \\
        --output-dir data/mes/filtrated
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.paths import get_default_paths
from src.noninformative.features import BottleneckExtractor, extract_all
from src.noninformative.model import NonInformativeClassifier

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_EPSILON = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter frames with informative RF model (ulcer and MES pipelines)."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory of processed frames."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to write kept frames."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(get_default_paths().informative_model_path),
        help="Path to rf_pipeline.pkl.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Uncertainty threshold |prob-0.5| < ε → uncertain (default: 0.0).",
    )
    return parser


def _scan_frames(input_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            records.append({"image_path": path, "rel_path": path.relative_to(input_dir)})
    return records


def _extract_features(
    records: list[dict], groups: list[str] | None, use_bottleneck: bool
) -> np.ndarray:
    paths = [rec["image_path"] for rec in records]
    extractor = BottleneckExtractor() if use_bottleneck else None
    return extract_all(
        paths,
        use_bottleneck=use_bottleneck,
        bottleneck_extractor=extractor,
        verbose=True,
        groups=groups,
    )


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    records = _scan_frames(input_dir)
    if not records:
        raise RuntimeError(f"No frames found in: {input_dir}")

    model = NonInformativeClassifier.load(model_path)

    cache_file = get_default_paths().informative_features_cache
    if not cache_file.exists():
        raise FileNotFoundError(f"Training features cache not found: {cache_file}")
    with open(cache_file, "rb") as fh:
        cache = pickle.load(fh)
    groups = cache.get("groups")
    use_bottleneck = cache.get("use_bottleneck", True)

    X = _extract_features(records, groups=groups, use_bottleneck=use_bottleneck)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= model.threshold).astype(int)
    uncertain_mask = np.abs(probs - 0.5) < args.epsilon

    rows = []
    kept = rejected = uncertain = 0

    for rec, prob, pred, is_uncertain in tqdm(
        zip(records, probs, preds, uncertain_mask),
        total=len(records),
        desc="Filter frames",
        unit="img",
    ):
        if is_uncertain:
            category = "uncertain"
            uncertain += 1
        elif pred == 1:
            category = "informative"
            kept += 1
            dst = output_dir / rec["rel_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec["image_path"], dst)
        else:
            category = "non_informative"
            rejected += 1

        rows.append(
            {
                "image_path": str(rec["image_path"]),
                "relative_path": str(rec["rel_path"]),
                "pred_prob": float(prob),
                "pred_label": int(pred),
                "category": category,
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)

    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "total_frames": len(records),
        "kept_informative": kept,
        "rejected_non_informative": rejected,
        "uncertain": uncertain,
        "kept_ratio": round(kept / max(len(records), 1), 4),
        "epsilon": float(args.epsilon),
        "threshold": float(model.threshold),
    }
    with open(output_dir / "filter_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print("=" * 72)
    print("FILTERING DONE")
    print("=" * 72)
    print(f"Input frames      : {len(records)}")
    print(f"Kept informative  : {kept}")
    print(f"Rejected          : {rejected}")
    print(f"Uncertain         : {uncertain}")
    print(f"Output dir        : {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main(build_parser().parse_args())
