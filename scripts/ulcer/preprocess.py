"""Run the full staged ulcer preprocessing pipeline.

Stages:
1) ROI preprocess raw frames into data/ulcer/processed
2) informative filtering into data/ulcer/filtrated
3) create train/val/test manifest from filtrated frames
4) generate EDA report

Usage
-----
    python scripts/ulcer/preprocess.py
    python scripts/ulcer/preprocess.py --skip-preprocess        # skip stage 1
    python scripts/ulcer/preprocess.py --incremental            # stage 1 skips existing frames
    python scripts/ulcer/preprocess.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scripts.data.preprocess_frames import main as preprocess_frames_main
from scripts.noninformative.filter_frames import main as filter_main
from scripts.ulcer.create_manifest import main as create_manifest_main
from scripts.ulcer.eda import main as eda_main
from src.config.paths import UlcerPaths, get_default_paths
from src.data.splits import STRAT_MODES


def build_parser() -> argparse.ArgumentParser:
    cfg = get_default_paths()
    parser = argparse.ArgumentParser(
        description="Run staged ulcer preprocessing pipeline.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(cfg.ulcer.root),
        help="Root data directory for this dataset (default: data/ulcer). "
             "All stage dirs (raw, processed, filtrated, splits) are derived from it.",
    )
    parser.add_argument(
        "--timestamps-file",
        type=str,
        default=None,
        help="Path to the timestamps workbook (default: <root>/raw/Ulcer and Non-Ulcer Timestamps.xlsx).",
    )
    parser.add_argument("--timestamps-sheet", type=str, default="Ulcer timestamps")
    parser.add_argument(
        "--model",
        type=str,
        default=str(cfg.informative_model_path),
        help="Informative filtering model path.",
    )
    parser.add_argument(
        "--eda-output-dir",
        type=str,
        default=str(cfg.results_eda_dir),
        help="Directory where EDA report is written.",
    )
    parser.add_argument(
        "--olympus-mask-path",
        type=str,
        default=str(cfg.results_ulcer_dir / "mask_olympus.png"),
        help="Path where the Olympus reference mask is saved for verification.",
    )
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strat-mode",
        choices=STRAT_MODES,
        default="ulcer_ratio",
        help="Patient stratification strategy for manifest creation (default: ulcer_ratio).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip ROI preprocessing and start directly from stage 2 (informative filtering).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip frames already present in processed-dir (stage 1 only).",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force informative-filter feature re-extraction, ignoring "
             "results/ulcer/filtering/features_cache.pkl (stage 2 only).",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    # Derive all stage directories from the dataset root
    ulcer      = UlcerPaths(root=Path(args.root))
    timestamps = args.timestamps_file or str(ulcer.raw / "Ulcer and Non-Ulcer Timestamps.xlsx")

    print("=" * 72)
    print("ULCER STAGED PREPROCESSING")
    print(f"  Root : {ulcer.root}")
    print("=" * 72)

    if not args.skip_preprocess:
        if not args.incremental:
            if ulcer.processed.exists():
                shutil.rmtree(ulcer.processed)
                print(f"Cleared processed dir: {ulcer.processed}")
        preprocess_frames_main(
            argparse.Namespace(
                raw_dir=str(ulcer.raw),
                output_dir=str(ulcer.processed),
                olympus_mask_path=args.olympus_mask_path,
                jpeg_quality=args.jpeg_quality,
                incremental=args.incremental,
            )
        )

    model_path = Path(args.model)

    if not model_path.exists():
        print(
            f"\n  [!] Skipping filtration — file not found: {model_path}"
            "\n  To enable filtration, place a pretrained rf_pipeline.pkl under "
            "data/assets/informative/ (features_cache.pkl is optional — only "
            "needed if the model's feature_names weren't saved)."
        )
    else:
        if ulcer.filtrated.exists():
            shutil.rmtree(ulcer.filtrated)
            print(f"Cleared filtrated dir: {ulcer.filtrated}")
        try:
            filter_main(
                argparse.Namespace(
                    input_dir=str(ulcer.processed),
                    output_dir=str(ulcer.filtrated),
                    model=args.model,
                    epsilon=args.epsilon,
                    recompute=args.recompute,
                )
            )
        except Exception as exc:
            print(
                f"\n  [!] Filtration failed (incompatible model/feature config?): {exc}\n"
                "  Skipping — retrain the filter or check features_cache.pkl if present."
            )

    create_manifest_main(
        argparse.Namespace(
            input_dir=str(ulcer.filtrated),
            splits_dir=str(ulcer.splits),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            strat_mode=args.strat_mode,
        )
    )

    eda_main(
        argparse.Namespace(
            raw_dir=str(ulcer.raw),
            processed_dir=str(ulcer.processed),
            filtrated_dir=str(ulcer.filtrated),
            splits_dir=str(ulcer.splits),
            output_dir=args.eda_output_dir,
            excel=timestamps,
            fps=10.0,
            image_stats=False,
            image_sample_size=None,
        )
    )

    print("=" * 72)
    print("ULCER PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Root             : {ulcer.root}")
    print(f"Raw frames       : {ulcer.raw}")
    print(f"Processed frames : {ulcer.processed}")
    print(f"Filtrated frames : {ulcer.filtrated}")
    print(f"Splits           : {ulcer.splits}")
    print(f"EDA              : {args.eda_output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main(build_parser().parse_args())
