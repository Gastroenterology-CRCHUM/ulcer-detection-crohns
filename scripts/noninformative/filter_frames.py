"""Generic RF informative-frame filter — reusable for any frame dataset.

Scans input_dir recursively, classifies frames with the informative RF model,
and copies informative frames to output_dir preserving relative path structure.

Usage
-----
    python scripts/noninformative/filter_frames.py \\
        --input-dir data/ulcer/processed \\
        --output-dir data/ulcer/filtrated
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.paths import get_default_paths
from src.noninformative.features import BottleneckExtractor, extract_all, infer_feature_config
from src.noninformative.model import NonInformativeClassifier

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_EPSILON = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter frames with informative RF model."
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
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force feature re-extraction, ignoring any cached results/ulcer/filtering/features_cache.pkl.",
    )
    return parser


def _scan_frames(input_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            records.append({"image_path": path, "rel_path": path.relative_to(input_dir)})
    return records


def _frames_fingerprint(input_dir: Path, records: list[dict]) -> str:
    """Hash of (path, size, mtime) for every scanned frame — stable iff the
    frame set and its content are unchanged, so a cached feature matrix can
    be reused across repeated runs on an unchanged input_dir."""
    parts = [str(input_dir.resolve())]
    for rec in records:
        st = rec["image_path"].stat()
        parts.append(f"{rec['rel_path']}|{st.st_size}|{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _extract_features(
    records: list[dict],
    groups: list[str] | None,
    use_handcrafted: bool,
    use_bottleneck: bool,
) -> np.ndarray:
    paths = [rec["image_path"] for rec in records]
    extractor = BottleneckExtractor() if use_bottleneck else None
    return extract_all(
        paths,
        use_handcrafted=use_handcrafted,
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
    if cache_file.exists():
        with open(cache_file, "rb") as fh:
            cache = pickle.load(fh)
        groups = cache.get("groups")
        use_handcrafted = cache.get("use_handcrafted", True)
        use_bottleneck = cache.get("use_bottleneck", True)
    elif model.feature_names:
        groups, use_handcrafted, use_bottleneck = infer_feature_config(model.feature_names)
        print(
            f"  [info] {cache_file} not found — inferred feature config from "
            f"rf_pipeline.pkl: groups={groups}, use_handcrafted={use_handcrafted}, "
            f"use_bottleneck={use_bottleneck}"
        )
    else:
        groups, use_handcrafted, use_bottleneck = None, True, True
        print(
            f"  [warn] {cache_file} not found and rf_pipeline.pkl has no feature_names — "
            "assuming all hand-crafted groups + bottleneck; this may not match the trained model."
        )

    filtering_cache_path = get_default_paths().results_filtering_dir / "features_cache.pkl"
    fingerprint = _frames_fingerprint(input_dir, records)

    X = None
    if not getattr(args, "recompute", False) and filtering_cache_path.exists():
        with open(filtering_cache_path, "rb") as fh:
            feat_cache = pickle.load(fh)
        if (
            feat_cache.get("fingerprint") == fingerprint
            and feat_cache.get("groups") == groups
            and feat_cache.get("use_handcrafted") == use_handcrafted
            and feat_cache.get("use_bottleneck") == use_bottleneck
        ):
            print(f"  [cache] Frames + config unchanged — reusing cached features ← {filtering_cache_path}")
            X = feat_cache["X"]
        else:
            print("  [cache] Frames or feature config changed — re-extracting.")

    if X is None:
        X = _extract_features(
            records, groups=groups, use_handcrafted=use_handcrafted, use_bottleneck=use_bottleneck
        )
        filtering_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(filtering_cache_path, "wb") as fh:
            pickle.dump(
                {
                    "fingerprint": fingerprint,
                    "input_dir": str(input_dir),
                    "groups": groups,
                    "use_handcrafted": use_handcrafted,
                    "use_bottleneck": use_bottleneck,
                    "X": X,
                },
                fh,
            )
        print(f"  [cache] Features cached → {filtering_cache_path}")

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
