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
│   │   └── Ulcer and Non-Ulcer Timestamps.xlsx
│   ├── processed/                 # ROI-cropped frames (1350×1080)
│   │   ├── Ulcer/
│   │   └── NonUlcer/
│   ├── filtrated/                 # Informative-only frames after RF filter
│   │   ├── Ulcer/
│   │   └── NonUlcer/
│   └── splits/                    # Train/val/test manifests
│       ├── dataset_manifest.csv
│       ├── split_info.json
│       └── heldout_temporal_manifest.csv  ← not included (IRB)
│
├── informative/                   # Informative-frame RF classifier data
│   ├── raw/
│   │   ├── Informative/
│   │   └── Non-Informative/
│   ├── processed/
│   └── splits/
│
└── assets/
    ├── pretrained/                # GastroNet weight files (download separately)
    │   ├── RN50_GastroNet-5M_DINOv1.pth
    │   └── VITS_GastroNet-5M_DINOv1.pth
    └── informative/               # Trained RF classifier artifacts
        ├── rf_pipeline.pkl
        └── features_cache.pkl

output/
├── ulcer/
│   └── models/
│       └── detection/
│           └── {model}/{timestamp}/best.pt
└── informative/
    └── models/

results/
└── ulcer/
    ├── cv/                        # CV result figures and tables
    └── eda/                       # EDA figures and reports
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

# Informative pipeline
paths.informative.raw    # data/informative/raw
paths.informative.splits # data/informative/splits

# Convenience aliases
paths.ulcer_splits_dir   # data/ulcer/splits
paths.results_eda_dir    # results/ulcer/eda
paths.results_cv_dir     # results/ulcer/cv
```

## Preprocessing Flow

```
videos/ + Timestamps.xlsx
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
