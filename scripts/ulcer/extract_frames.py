"""Extract and subsample ulcer frames from annotated videos.

For each annotated segment (record_id, start_s, end_s) read from --excel,
samples frames from the matching video at --fps and, when a segment yields
more than --max-frames frames, keeps only the --max-frames most visually
diverse ones (GastroNet backbone embeddings, greedy farthest-point sampling),
 the discarded frames are deleted, not just excluded from the manifest.

Informative-frame filtering is NOT done here: the RF filter needs ROI-cropped
frames to match its normal operating point, so it runs later in the pipeline,
on data/ulcer/processed/, via scripts/noninformative/filter_frames.py (see
scripts/ulcer/preprocess.py). The diversity subsampling above still crops
each frame in-memory (per detected Fuji/Olympus platform) before computing
its embedding, purely to keep the endoscope UI panel from dominating the
similarity signal, the frame files written to --out-dir stay uncropped.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.paths import get_default_paths
from src.data.annotation_loaders import load_ulcer_annotations
from src.data.subsampling import load_backbone_for_embeddings, visual_subsample
from src.data.video_extraction import build_video_index_generic, extract_frames_from_video
from src.data.video_utils import _detect_platform_from_video, crop_platform, find_overlay_offset

logger = logging.getLogger(__name__)

LABEL_DIRS = {1: "Ulcer", 0: "NonUlcer"}


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------


def _process_frames(
    frames: list[Path],
    *,
    backbone,
    max_frames: int | None,
    device: str,
    crop_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Path]:
    """Keep up to max_frames per segment, deleting the discarded frames from disk."""
    if max_frames is None or len(frames) <= max_frames:
        return frames
    kept = visual_subsample(
        frames, max_frames, backbone=backbone, device=device, preprocess_fn=crop_fn
    )
    for f in set(frames) - set(kept):
        f.unlink(missing_ok=True)
    return kept


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
    if not video_index:
        raise ValueError(f"No video files found under --input: {input_path}")

    out_dir = Path(args.out_dir)
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
            logger.warning("No video for record_id=%s, skipped.", record_id)
            continue

        platform = _detect_platform_from_video(video_path)
        crop_fn = (lambda img: crop_platform(img, platform)) if backbone is not None else None

        offset_s = 0.0
        if not args.no_ocr_offset:
            ocr_offset = find_overlay_offset(
                video_path,
                tuple(args.ocr_probe_times),
                Path(args.ocr_debug_dir) if args.ocr_debug_dir else None,
                is_fuji=(platform == "fuji"),
            )
            if ocr_offset is not None:
                offset_s = ocr_offset
            else:
                logger.warning("OCR offset not found for %s, using 0.0.", record_id)

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
                backbone=backbone,
                max_frames=args.max_frames,
                device=args.device,
                crop_fn=crop_fn,
            )
            for f in frames:
                manifest_rows.append(
                    {
                        "relative_path": str(f.relative_to(out_dir)),
                        "label": label,
                        "record_id": record_id,
                        "sample_number": sample_num,
                    }
                )

    if skipped:
        logger.warning("%d record(s) skipped, no matching video.", len(skipped))

    return pd.DataFrame(manifest_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    paths = get_default_paths()
    parser = argparse.ArgumentParser(
        description="Extract and subsample ulcer frames from annotated videos."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Video file or directory of video files.",
    )
    parser.add_argument(
        "--excel",
        default=str(paths.ulcer_raw_dir / "annotations.xlsx"),
        help="Annotations Excel workbook.",
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

    manifest = _run_video_mode(args)

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
