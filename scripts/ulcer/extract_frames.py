"""Extract, filter, and subsample ulcer frames from annotated videos or pre-extracted frames.

Input modes (auto-detected from --input):
  video   --input is a video file or a directory containing video files
  frames  --input is a directory of pre-extracted frames (no videos found)

Video mode requires --excel.  Frame mode can run without --excel.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.paths import get_default_paths
from src.data.annotation_loaders import load_ulcer_annotations
from src.data.video_utils import _detect_platform_from_video, find_overlay_offset
from src.data.subsampling import load_backbone_for_embeddings, visual_subsample
from src.data.video_extraction import (
    VIDEO_EXTS,
    build_video_index_generic,
    collect_frames_from_dir,
    extract_frames_from_video,
)
from src.noninformative.features import BottleneckExtractor, extract_all
from src.noninformative.model import NonInformativeClassifier

logger = logging.getLogger(__name__)

LABEL_DIRS = {1: "Ulcer", 0: "NonUlcer"}


# ---------------------------------------------------------------------------
# Informative filter
# ---------------------------------------------------------------------------


def _make_informative_filter(
    model_path: Path,
    features_cache_path: Path,
) -> Callable[[list[Path]], list[Path]] | None:
    """Load informative RF classifier and return a frame-list filter function.

    Returns None if the model file does not exist.
    """
    if not model_path.exists():
        logger.warning("Informative model not found at %s — filtering disabled.", model_path)
        return None

    model = NonInformativeClassifier.load(model_path)

    groups = None
    use_bottleneck = True
    if features_cache_path.exists():
        with open(features_cache_path, "rb") as fh:
            cache = pickle.load(fh)
        groups = cache.get("groups")
        use_bottleneck = cache.get("use_bottleneck", True)

    extractor = BottleneckExtractor() if use_bottleneck else None

    def filter_fn(frame_paths: list[Path]) -> list[Path]:
        if not frame_paths:
            return []
        images_rgb = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            images_rgb.append(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is not None
                else np.zeros((224, 224, 3), dtype=np.uint8)
            )
        X = extract_all(
            images_rgb,
            use_bottleneck=use_bottleneck,
            bottleneck_extractor=extractor,
            groups=groups,
            verbose=False,
        )
        probs = model.predict_proba(X)[:, 1]
        return [p for p, prob in zip(frame_paths, probs) if prob >= model.threshold]

    logger.info("Informative filter loaded from %s.", model_path.name)
    return filter_fn


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------


def _process_frames(
    frames: list[Path],
    *,
    filter_fn: Callable[[list[Path]], list[Path]] | None,
    backbone,
    max_frames: int | None,
    device: str,
) -> list[Path]:
    if filter_fn is not None:
        frames = filter_fn(frames)
    if max_frames is not None and len(frames) > max_frames:
        frames = visual_subsample(frames, max_frames, backbone=backbone, device=device)
    return frames


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------


def _run_video_mode(args: argparse.Namespace) -> pd.DataFrame:
    excel_path = Path(args.excel)
    ann_df = load_ulcer_annotations(excel_path)
    logger.info("Annotations: %d segments loaded.", len(ann_df))

    input_path = Path(args.input)
    video_index = (
        {input_path.stem.lower(): input_path}
        if input_path.is_file()
        else build_video_index_generic(input_path)
    )
    logger.info("Video index: %d file(s).", len(video_index))

    out_dir = Path(args.out_dir)
    filter_fn = (
        None
        if args.no_informative_filter
        else _make_informative_filter(
            Path(args.informative_model), Path(args.informative_features_cache)
        )
    )
    backbone = (
        None
        if args.no_subsample
        else load_backbone_for_embeddings(
            arch=args.backbone_arch,
            checkpoint_path=Path(args.backbone_checkpoint) if args.backbone_checkpoint else None,
            device=args.device,
        )
    )

    manifest_rows: list[dict] = []
    skipped: list[dict] = []

    for record_id, rows in tqdm(ann_df.groupby("record_id"), desc="Records", unit="record"):
        video_path = video_index.get(str(record_id).lower())
        if video_path is None:
            skipped.append({"record_id": record_id, "reason": "video_not_found"})
            logger.warning("No video for record_id=%s — skipped.", record_id)
            continue

        offset_s = 0.0
        if not args.no_ocr_offset:
            is_fuji = _detect_platform_from_video(video_path) == "fuji"
            ocr_offset = find_overlay_offset(
                video_path,
                tuple(args.ocr_probe_times),
                Path(args.ocr_debug_dir) if args.ocr_debug_dir else None,
                is_fuji=is_fuji,
            )
            if ocr_offset is not None:
                offset_s = ocr_offset
            else:
                logger.warning("OCR offset not found for %s — using 0.0.", record_id)

        for _, row in rows.iterrows():
            label = int(row["label"])
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            sample_num = int(row.get("sample_number") or 0)

            segment_dir = (
                out_dir
                / LABEL_DIRS.get(label, f"label_{label}")
                / str(record_id)
                / f"sample_{sample_num:02d}"
            )
            frames = extract_frames_from_video(
                video_path,
                start_s,
                end_s,
                segment_dir,
                fps_target=args.fps,
                offset_s=offset_s,
                jpeg_quality=args.jpeg_quality,
                frame_prefix=f"{record_id}__s{sample_num:02d}",
                skip_existing=args.skip_existing,
            )
            frames = _process_frames(
                frames,
                filter_fn=filter_fn,
                backbone=backbone,
                max_frames=args.max_frames,
                device=args.device,
            )
            for f in frames:
                manifest_rows.append(
                    {
                        "relative_path": str(f.relative_to(out_dir)),
                        "label": label,
                        "record_id": record_id,
                        "sample_number": sample_num,
                        "size": row.get("size"),
                    }
                )

    if skipped:
        logger.warning("%d record(s) skipped — no matching video.", len(skipped))

    return pd.DataFrame(manifest_rows)


# ---------------------------------------------------------------------------
# Frames mode
# ---------------------------------------------------------------------------


def _run_frames_mode(args: argparse.Namespace) -> pd.DataFrame:
    frames_dir = Path(args.input)
    out_dir = Path(args.out_dir)

    filter_fn = (
        None
        if args.no_informative_filter
        else _make_informative_filter(
            Path(args.informative_model), Path(args.informative_features_cache)
        )
    )
    backbone = (
        None
        if args.no_subsample
        else load_backbone_for_embeddings(
            arch=args.backbone_arch,
            checkpoint_path=Path(args.backbone_checkpoint) if args.backbone_checkpoint else None,
            device=args.device,
        )
    )

    # Leaf directories = dirs containing image files
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    leaf_dirs = [
        d
        for d in sorted(frames_dir.rglob("*"))
        if d.is_dir() and any(f.suffix.lower() in image_exts for f in d.iterdir() if f.is_file())
    ]

    manifest_rows: list[dict] = []

    for segment_dir in tqdm(leaf_dirs, desc="Segments", unit="dir"):
        frames = collect_frames_from_dir(segment_dir)
        frames = _process_frames(
            frames,
            filter_fn=filter_fn,
            backbone=backbone,
            max_frames=args.max_frames,
            device=args.device,
        )
        parts = segment_dir.relative_to(frames_dir).parts
        label_name = parts[0] if parts else "unknown"
        label = 1 if label_name == "Ulcer" else 0
        base = out_dir if args.out_dir != args.input else frames_dir
        for f in frames:
            manifest_rows.append({"relative_path": str(f.relative_to(base)), "label": label})

    return pd.DataFrame(manifest_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    paths = get_default_paths()
    parser = argparse.ArgumentParser(
        description="Extract and subsample ulcer frames from annotated videos or pre-extracted frames."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Video file, directory of videos, or directory of pre-extracted frames.",
    )
    parser.add_argument(
        "--excel",
        default=str(paths.ulcer_raw_dir / "Ulcer and Non-Ulcer Timestamps.xlsx"),
        help="Annotations Excel (required in video mode).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(paths.ulcer_raw_dir),
        help="Output directory for extracted frames.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second to extract.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Max frames per segment after subsampling (0 = no limit).",
    )
    parser.add_argument("--no-subsample", action="store_true", help="Disable visual subsampling.")
    parser.add_argument("--no-informative-filter", action="store_true")
    parser.add_argument(
        "--informative-model",
        default=str(paths.informative_model_path),
        help="Path to informative RF classifier pickle.",
    )
    parser.add_argument(
        "--informative-features-cache",
        default=str(paths.informative_features_cache),
        help="Path to features cache pickle (for groups / bottleneck settings).",
    )
    parser.add_argument(
        "--backbone-arch",
        default="resnet50_gastronet",
        help="GastroNet architecture key for visual subsampling embeddings.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="Optional .pt checkpoint to override pretrained GastroNet backbone weights.",
    )
    parser.add_argument("--no-ocr-offset", action="store_true")
    parser.add_argument("--ocr-probe-times", type=int, nargs="+", default=[120, 180, 240])
    parser.add_argument("--ocr-debug-dir", type=str, default=None)
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda).")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.max_frames == 0:
        args.max_frames = None

    input_path = Path(args.input)
    has_videos = input_path.is_file() or any(
        p.suffix.lower() in VIDEO_EXTS for p in input_path.rglob("*") if p.is_file()
    )
    manifest = _run_video_mode(args) if has_videos else _run_frames_mode(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "extracted_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("=" * 60)
    print("ULCER EXTRACTION DONE")
    print("=" * 60)
    print(f"Frames    : {len(manifest)}")
    print(f"Output    : {out_dir}")
    print(f"Manifest  : {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main(build_parser().parse_args())
