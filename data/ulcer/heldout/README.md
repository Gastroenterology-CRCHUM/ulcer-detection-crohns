# Held-out temporal test set

Lives separately from `data/ulcer/splits/` because it isn't one of the
train/val/test splits of the main cohort, it's an independent patient
cohort acquired after the training cohort's cutoff date, used only for
final held-out evaluation.

The raw video data cannot be shared publicly due to IRB restrictions
(CER 22.013, NCT06822816), this directory (frames + manifest) is not
included in the public repository either, but is regenerated locally by
whoever already has legitimate access to the frames.

## Structure

```
heldout/
├── Ulcer/<video_id>/<segment>/*.jpg       # canonical set
├── NonUlcer/<video_id>/<segment>/*.jpg
├── heldout_temporal_manifest.csv          # generated
└── raw/{Ulcer,NonUlcer}/...               # raw frames
```

## Preprocess frames

```bash
python -m scripts.data.preprocess_frames \
    --raw-dir data/ulcer/heldout/raw \
    --output-dir data/ulcer/heldout
```

Generates the canonical set (top-level `Ulcer/` + `NonUlcer/`) composed of the ROI-cropped frames. It matches the 1,573 frames / 65 clips (37 ulcer-positive, 28 ulcer-negative) / 19 patients used to produce Tables 1 and 2 in the paper.

## Generating heldout_temporal_manifest.csv

Once `Ulcer/` and `NonUlcer/` are populated (as above):

```bash
python -m scripts.ulcer.create_heldout_manifest
```

Scans `data/ulcer/heldout/{Ulcer,NonUlcer}` and writes `heldout_temporal_manifest.csv` with every row's `split` set to `"heldout"`, unlike `scripts/ulcer/create_manifest.py`, there is no train/val/test splitting involved.

### CSV format

```
relative_path,video_id,patient_id,clip_key,label,split
Ulcer/vid_XX_YYYY/ulcer_1/frame_000.jpg,vid_XX_YYYY,vid_XX_YYYY,vid_XX_YYYY__ulcer_1,1,heldout
NonUlcer/vid_XX_ZZZZ/normal_1/frame_000.jpg,vid_XX_ZZZZ,vid_XX_ZZZZ,vid_XX_ZZZZ__normal_1,0,heldout
...
```

`relative_path` is relative to `data/ulcer/heldout/` (the same directory passed as `--heldout-data-dir` / `--data-dir` below).

## Reproducing held-out evaluation

```bash
python scripts/run_experiments.py \
    --plan configs/experiments/ulcer_batch.yaml \
    --heldout-manifest data/ulcer/heldout/heldout_temporal_manifest.csv
    # --heldout-data-dir defaults to data/ulcer/heldout, pass it explicitly to override

# Then compute clip-level metrics across all folds:
python -m scripts.ulcer.log_heldout_clip_metrics \
    --manifest data/ulcer/heldout/heldout_temporal_manifest.csv

# And best-fold CIs (Table 2):
python -m scripts.ulcer.log_heldout_clip_best_fold \
    --manifest data/ulcer/heldout/heldout_temporal_manifest.csv
```
