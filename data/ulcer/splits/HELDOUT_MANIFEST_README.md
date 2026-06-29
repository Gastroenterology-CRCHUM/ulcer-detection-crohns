# Held-out temporal test set

The file `heldout_temporal_manifest.csv` is not included in this repository.

It contains the temporal held-out test set used to produce Tables 1 and 2
in the paper: 1,573 frames, 65 clips (37 ulcer-positive, 28 ulcer-negative),
from 19 patients acquired after the training cohort cutoff.

The raw video data cannot be shared publicly due to IRB restrictions
(CER 22.013, NCT06822816).

## Expected CSV format

```
relative_path,video_id,patient_id,clip_key,label,split
Ulcer/vid_XX_YYYY/ulcer_1/frame_000.jpg,vid_XX_YYYY,vid_XX_YYYY,vid_XX_YYYY__ulcer_1,1,heldout
NonUlcer/vid_XX_ZZZZ/normal_1/frame_000.jpg,vid_XX_ZZZZ,vid_XX_ZZZZ,vid_XX_ZZZZ__normal_1,0,heldout
...
```

## Reproducing held-out evaluation

Once you have the manifest, run:

```bash
python scripts/run_experiments.py \
    --plan configs/experiments/ulcer_batch.yaml \
    --heldout-manifest data/ulcer/splits/heldout_temporal_manifest.csv

# Then compute clip-level metrics across all folds:
python -m scripts.ulcer.log_heldout_clip_metrics \
    --manifest data/ulcer/splits/heldout_temporal_manifest.csv

# And best-fold CIs (Table 2):
python -m scripts.ulcer.log_heldout_clip_best_fold \
    --manifest data/ulcer/splits/heldout_temporal_manifest.csv
```
