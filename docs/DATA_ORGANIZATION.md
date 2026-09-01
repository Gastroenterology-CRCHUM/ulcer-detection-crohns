# Data Organization

## Directory Structure

```
data/
├── ulcer/
│   ├── raw/                       # Source frames extracted from annotated videos
│   │   ├── Ulcer/
│   │   │   └── vid_XX_XXXX/
│   │   │       └── ulcer_X/
│   │   │           └── *.jpg
│   │   ├── NonUlcer/
│   │   │   └── vid_XX_XXXX/
│   │   │       └── normal_X/
│   │   │           └── *.jpg
│   │   ├── videos/                # Original .mov/.mp4 files
│   │   └── annotations.xlsx       # sheets: "ulcer", "non_ulcer"
│   ├── processed/                 # ROI-cropped frames (1350×1080)
│   │   ├── Ulcer/
│   │   └── NonUlcer/
│   ├── filtrated/                 # Informative-only frames after RF filter
│   │   ├── Ulcer/
│   │   └── NonUlcer/
│   ├── splits/                    # Train/val/test manifests (main cohort)
│   │   ├── dataset_manifest.csv
│   │   └── split_info.json
│   └── heldout/                   # Temporal held-out test cohort, a separate patient
│       │                          # cohort acquired after the training cutoff.
│       ├── README.md
│       ├── Ulcer/, NonUlcer/      # canonical set: ROI-cropped
│       ├── heldout_temporal_manifest.csv  # generated (see create_heldout_manifest.py)
│       ├── raw/                   # Pre-crop frames
│       └── filtrated/             # archival: RF-filtered variant, unused
│
└── assets/
    ├── pretrained/                # GastroNet weight files (download separately)
    │   ├── RN50_GastroNet-5M_DINOv1.pth
    │   └── VITS_GastroNet-5M_DINOv1.pth
    └── informative/               # Pretrained RF classifier artifacts (filtering only)
        └── rf_pipeline.pkl

output/
└── ulcer/
    └── models/
        └── detection/
            └── {model}/{timestamp}/best.pt

results/
├── ulcer/
│   ├── cv/                        # CV result figures and tables
│   ├── filtering/
│   │   └── features_cache.pkl     # Cache to speed up image features extraction before filtering
│   └── eda/                       # EDA figures and reports
└── mask_olympus.png               # saving the mask for ROI crop
```

## Path Management

All paths are centralized in `src/config/paths.py` via the `PathConfig` dataclass.

```python
from src.config.paths import get_default_paths

paths = get_default_paths()

# Ulcer pipeline
paths.ulcer.raw          # data/ulcer/raw
paths.ulcer.processed    # data/ulcer/processed
paths.ulcer.filtrated    # data/ulcer/filtrated
paths.ulcer.splits       # data/ulcer/splits
paths.ulcer.heldout      # data/ulcer/heldout

# Convenience aliases
paths.ulcer_splits_dir   # data/ulcer/splits
paths.ulcer_heldout_dir  # data/ulcer/heldout
paths.results_eda_dir    # results/ulcer/eda
paths.results_cv_dir     # results/ulcer/cv
```

## Preprocessing Flow

```
videos/ + annotations.xlsx
        ↓  scripts/ulcer/extract_frames.py
    raw/
        ↓  scripts/data/preprocess_frames.py  (ROI crop)
 processed/
        ↓  scripts/noninformative/filter_frames.py  (RF informative filter)
 filtrated/
        ↓  scripts/ulcer/create_manifest.py  (patient-stratified split)
   splits/dataset_manifest.csv
        ↓  scripts/ulcer/eda.py
  results/ulcer/eda/
```

`heldout/` is not part of this flow, it's an independent test cohort, evaluated post-hoc against already-trained models.
Once `heldout/{Ulcer,NonUlcer}` is populated, its manifest is generated with
`scripts/ulcer/create_heldout_manifest.py` (no train/val/test splitting,
every row gets `split="heldout"`). See `data/ulcer/heldout/README.md`.
