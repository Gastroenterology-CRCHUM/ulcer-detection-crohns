# Domain-Specific Foundation Models vs Conventional Deep Learning for Automated Ulcer Detection in Crohn's Disease

Code for the paper:

> **Domain-Specific Foundation Models Versus Conventional Deep Learning for Automated Ulcer Detection in Crohn's Disease**  
> Yann-Raphael Berndt\*, Nikoo Mashayekhi\*, Chelssy Guerine Ingabire, Robert Battat, Michael Byrne, Daniel von Renteln; AI-CD working group  
> \*Shared first authorship

## Overview

This repository contains the full pipeline for the study: from raw colonoscopy video extraction through informative-frame filtering, model training, cross-validation, and held-out evaluation.

Nine model configurations were evaluated by combining four architectures (ResNet-50, EfficientNet-B0, ViT-Small/16, ViT-Base/16) with up to three pretraining strategies (supervised ImageNet-1K, self-supervised DINOv1/ImageNet, self-supervised DINOv1/GastroNet-5M).

## Repository Layout

```text
.
├── configs/
│   ├── example.yaml                       Reference for all config fields and defaults
│   └── experiments/
│       └── ulcer_batch.yaml               Experiment plan — 9 model configurations from the paper
├── scripts/
│   ├── run_experiments.py                 Batch experiment orchestrator
│   ├── data/                              Preprocessing utilities (ROI crop, frame extraction)
│   ├── noninformative/                    Informative-frame RF filter (loads pretrained rf_pipeline.pkl)
│   └── ulcer/
│       ├── extract_frames.py              Frame extraction from annotated videos
│       ├── preprocess.py                  Full 4-stage preprocessing pipeline
│       ├── create_manifest.py             Build train/val/test manifest CSV
│       ├── eda.py                         Dataset EDA reports and figures
│       ├── train.py                       Single-run training (split or CV mode)
│       ├── evaluate_with_delong.py        Multi-model DeLong AUROC comparison
│       ├── log_heldout_clip_metrics.py    Clip-level held-out metrics (all CV folds)
│       ├── log_heldout_clip_best_fold.py  Clip-level held-out CI (best fold)
│       └── statistical_comparison.py      Friedman + Wilcoxon pairwise tests on CV fold AUROCs
├── src/
│   ├── config/                            Dataclass configuration and MODEL_REGISTRY
│   ├── data/                              Datasets, dataloaders, splits, transforms, extraction
│   ├── evaluation/                        Metrics, bootstrap CI, DeLong test, plots, MLflow helpers
│   ├── models/                            ClassifierModel backbone wrapper
│   ├── noninformative/                    Feature extraction and RF inference
│   └── training/                         Training loop, run_split_mode, run_cv_mode
├── results/
│   └── ulcer/
│       ├── cv/                            CV result figures and tables (Figures 1–3, Tables 1–2)
│       └── eda/                           Dataset EDA figures
├── data/
│   └── ulcer/
│       └── splits/
│           └── HELDOUT_MANIFEST_README.md  Instructions for obtaining the temporal held-out manifest
└── tests/                                 Unit tests
```

## Setup

```bash
conda create -n ulcer-detection python=3.10
conda activate ulcer-detection

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm transformers scikit-learn pandas numpy opencv-python pillow openpyxl tqdm matplotlib seaborn scipy scikit-image
pip install joblib mlflow pyyaml scikit-posthocs statsmodels pytesseract pytest pytest-cov
```

`pytesseract` also requires the Tesseract-OCR engine itself (used by default for OCR-based overlay-offset detection in `extract_frames.py`; pass `--no-ocr-offset` to skip it):

- **Windows**: install via the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) (auto-detected at `C:\Program Files\Tesseract-OCR\tesseract.exe`), or ensure `tesseract` is on `PATH`.
- **Linux/macOS**: `sudo apt install tesseract-ocr` / `brew install tesseract`.

Optional — `.mov` video support on Windows:

```bash
conda install -c conda-forge ffmpeg
```

## Data Preparation

### Step 1 — Extract frames from annotated videos

```bash
python -m scripts.ulcer.extract_frames \
    --input data/ulcer/raw/videos \
    --excel annotations.xlsx
```

Auto-detects Fuji/Olympus endoscope, estimates OCR overlay offset, applies the informative-frame RF filter, and runs visual-diversity subsampling (GastroNet backbone, greedy farthest-point).

### Step 2 — Run full staged preprocessing

```bash
python -m scripts.ulcer.preprocess
```

Runs four stages in sequence:

| Stage | Input → Output | Script |
|-------|----------------|--------|
| ROI crop | `data/ulcer/raw/` → `data/ulcer/processed/` | `scripts/data/preprocess_frames.py` |
| Informative filtering | `data/ulcer/processed/` → `data/ulcer/filtrated/` | `scripts/noninformative/filter_frames.py` |
| Manifest creation | `data/ulcer/filtrated/` → `data/ulcer/splits/` | `scripts/ulcer/create_manifest.py` |
| EDA report | all of the above → `results/ulcer/eda/` | `scripts/ulcer/eda.py` |

Options:

```bash
python -m scripts.ulcer.preprocess --skip-preprocess   # skip ROI crop (stage 1)
python -m scripts.ulcer.preprocess --incremental        # skip already-processed frames
python -m scripts.ulcer.preprocess --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

### Informative-frame classifier

This repo ships only the filtering side of the RF classifier — it loads the
pretrained `data/assets/informative/rf_pipeline.pkl` via
`NonInformativeClassifier.load()` and applies it in
`scripts/noninformative/filter_frames.py` and `scripts/ulcer/extract_frames.py`.
`NonInformativeClassifier` (in `src/noninformative/model.py`) still exposes
`.fit()` / `.evaluate()` for programmatic retraining, but there is no longer
a dedicated training script — this repo focuses on the ulcer-detection
pipeline.

`rf_pipeline.pkl` alone is enough to run filtration — `features_cache.pkl`
(the feature-extraction config: `groups` / `use_handcrafted` / `use_bottleneck`)
is only a convenience cache and is git-ignored (too large to commit, since it
also holds the full training feature matrices). If it's missing, both scripts
fall back to `infer_feature_config()` (`src/noninformative/features.py`),
which reconstructs the same config from the `feature_names` embedded in
`rf_pipeline.pkl` itself — see "Inspecting an already-trained
`rf_pipeline.pkl`" below. `features_cache.pkl` only matters for the
training-time convenience it was originally built for (caching
`X_train`/`X_val`/`X_test`) — not for filtration.

**`filter_frames.py` also caches its own extracted features**, at
`results/ulcer/filtering/features_cache.pkl` — a different file with the
same name as the one above, don't confuse the two. It stores the feature
matrix for whatever `--input-dir` was last filtered, keyed on a hash of each
frame's path/size/mtime plus the `groups`/`use_handcrafted`/`use_bottleneck`
config. Re-running filtration (e.g. re-running `preprocess.py`) reuses it
as long as `data/ulcer/processed` hasn't changed, skipping extraction
entirely; touch a frame or change the feature config and it re-extracts
automatically. It's git-ignored (local, can grow large). Pass `--recompute`
to force re-extraction.

**If `data/assets/informative/rf_pipeline.pkl` isn't present** (e.g. a fresh
clone without the pretrained assets), generate it from your own labelled
Informative / Non-Informative frame manifest — e.g.
`data/informative/splits/dataset_manifest.csv` with `image_path`, `label`,
`split` columns (this raw dataset isn't bundled with the repo):

```python
import pickle
import pandas as pd
from src.config.paths import get_default_paths
from src.noninformative.features import extract_all, get_feature_names
from src.noninformative.model import NonInformativeClassifier

GROUPS = ["glcm", "intensity"]   # default hand-crafted groups (see note below)
USE_HANDCRAFTED = True           # False → bottleneck-only, for large datasets

paths = get_default_paths()
manifest = pd.read_csv("data/informative/splits/dataset_manifest.csv")

def features_for(split):
    df = manifest[manifest["split"] == split]
    X = extract_all(df["image_path"].tolist(), groups=GROUPS, use_handcrafted=USE_HANDCRAFTED)
    return X, df["label"].values

X_train, y_train = features_for("train")
X_val, y_val = features_for("val")
X_test, y_test = features_for("test")

feat_names = (get_feature_names(GROUPS) if USE_HANDCRAFTED else []) + [f"bn_{i}" for i in range(2048)]
clf = NonInformativeClassifier().fit(X_train, y_train, feature_names=feat_names)
clf.tune_threshold(X_val, y_val)
clf.evaluate(X_test, y_test)
clf.save(paths.informative_model_path)

with open(paths.informative_features_cache, "wb") as f:
    pickle.dump({"groups": GROUPS, "use_handcrafted": USE_HANDCRAFTED, "use_bottleneck": True}, f)
```

`filter_frames.py` and `extract_frames.py` both read `groups` /
`use_handcrafted` / `use_bottleneck` back from `features_cache.pkl` when
it's present, falling back to `infer_feature_config()` otherwise — keep the
cache in sync whenever you retrain with different settings, so the inferred
fallback doesn't have to guess.

**Manually inspecting an `rf_pipeline.pkl`** — same idea as
`infer_feature_config()`, as a standalone snippet:

```python
import pickle

with open("data/assets/informative/rf_pipeline.pkl", "rb") as f:
    state = pickle.load(f)

names = state["feature_names"]  # None if the model was trained without passing feature_names
if names:
    bn_names = [n for n in names if n.startswith("bn_")]
    print(f"{len(names)} total — bottleneck: {'yes (' + str(len(bn_names)) + ')' if bn_names else 'no'}, "
          f"hand-crafted: {[n for n in names if not n.startswith('bn_')]}")
else:
    print(f"{state['rf'].n_features_in_} total features (no names saved)")
```

**Feature groups default to `glcm` + `intensity`** (24 features) — that's
what the shipped `rf_pipeline.pkl` was trained on, and it adds relatively
little overhead on top of the 2048 bottleneck features. The full
hand-crafted extractor (`src/noninformative/features.py`, 6 groups / 43
features) is per-frame CPU-bound OpenCV/scikit-image work that adds up fast
at scale. **For large datasets, set `USE_HANDCRAFTED = False`** and rely on
bottleneck features alone — skipping hand-crafted extraction entirely is the
biggest speed lever for large runs.

## Running Experiments

### Single run

```bash
python -m scripts.ulcer.train --mode cv    # 5-fold CV
python -m scripts.ulcer.train --mode split # train/val/test split
```

### Batch experiments (all 9 configurations)

```bash
python scripts/run_experiments.py \
    --plan configs/experiments/ulcer_batch.yaml \
    --heldout-manifest data/ulcer/splits/heldout_temporal_manifest.csv
```

> **Note:** The temporal held-out manifest is not included due to IRB restrictions.
> See `data/ulcer/splits/HELDOUT_MANIFEST_README.md` for access instructions.
> Cross-validation runs and CV-level results (fold means/std) are fully reproducible
> without it.

Dry-run to preview the plan without training:

```bash
python scripts/run_experiments.py --dry-run
```

Filter to a single model:

```bash
python scripts/run_experiments.py --plan configs/experiments/ulcer_batch.yaml --model vits16_gastronet
```

With a held-out test manifest evaluated at each fold:

```bash
python scripts/run_experiments.py \
    --plan configs/experiments/ulcer_batch.yaml \
    --heldout-manifest data/ulcer/splits/heldout_manifest.csv
```

#### YAML plan format

```yaml
runs:
  - model: vits16_gastronet       # MODEL_REGISTRY key (required)
    freeze_layers: 0              # 0=full fine-tuning | -1=freeze backbone | N=first N blocks
    lr: 1.0e-4
    batch_size: 64
    epochs: 100
    mode: cv                      # split | cv
    dropout_rate: 0.3
    weight_decay: 1.0e-2
    label_smoothing: 0.0
    n_splits: 5                   # CV folds when mode=cv
    register: false               # register in MLflow Model Registry
```

## Post-hoc Evaluation

### Clip-level metrics (all CV folds)

```bash
python -m scripts.ulcer.log_heldout_clip_metrics
```

### Clip-level CI (best fold)

```bash
python -m scripts.ulcer.log_heldout_clip_best_fold
```

### DeLong pairwise AUROC comparison

```bash
python -m scripts.ulcer.evaluate_with_delong \
    --run-id <MLflow CV parent run ID> \
    --manifest data/ulcer/splits/heldout_manifest.csv \
    --data-dir data/ulcer/filtrated
```

### Friedman + Wilcoxon statistical comparison

Reads per-fold validation AUROCs from MLflow and produces:
- `results/ulcer/cv/friedman_ranks.png` — mean model rank + Friedman χ² p-value
- `results/ulcer/cv/wilcoxon_pmatrix.png` — pairwise Wilcoxon signed-rank p-value heatmap

```bash
python -m scripts.ulcer.statistical_comparison

# Custom MLflow store or experiment name:
python -m scripts.ulcer.statistical_comparison \
    --mlflow-uri sqlite:///mlflow.db \
    --experiment ulcer_detection
```

## Models

All models are defined in `src/config/models.py`. The registry maps a short key to backbone,
weights source, and pretraining metadata. Pass any key via `--model` or in a YAML plan.

### Paper models (9 configurations)

| Key | Architecture | Pretraining | Method | Weights source |
|-----|-------------|-------------|--------|----------------|
| `resnet50_imagenet_sup` | ResNet-50 | ImageNet-1K | Supervised | torchvision (auto) |
| `resnet50_imagenet` | ResNet-50 | ImageNet | DINOv1 | torch.hub (auto) |
| `resnet50_gastronet` | ResNet-50 | GastroNet-5M | DINOv1 | local `.pth` file |
| `efficientnetb0` | EfficientNet-B0 | ImageNet-1K | Supervised | torchvision (auto) |
| `vitb16_imagenet_sup` | ViT-Base/16 | ImageNet-1K | Supervised | torchvision (auto) |
| `vitb16_imagenet` | ViT-Base/16 | ImageNet | DINOv1 | torch.hub (auto) |
| `vits16_imagenet_hf` | ViT-Small/16 | ImageNet-1K | Supervised | timm (auto) |
| `vits16_imagenet` | ViT-Small/16 | ImageNet | DINOv1 | torch.hub (auto) |
| `vits16_gastronet` | ViT-Small/16 | GastroNet-5M | DINOv1 | local `.pth` file |

Models marked "auto" download their weights on first use. GastroNet models require
local weight files — see [Data Preparation](#data-preparation) below.

### GastroNet weight files

Download from the [GastroNet-5M paper](https://doi.org/10.1053/j.gastro.2025.07.030)
Weights available [here](https://cortex.thetavision.nl/dataset-provider/listing/2/): 
and place in `data/assets/pretrained/`:

| File | Required by |
|------|-------------|
| `RN50_GastroNet-5M_DINOv1.pth` | `resnet50_gastronet` |
| `VITS_GastroNet-5M_DINOv1.pth` | `vits16_gastronet` |

### Key training options

- `freeze_layers`: `0` = full fine-tuning (default), `-1` = frozen backbone, `N` = freeze first N encoder blocks
- `num_classes`: `1` = sigmoid output with per-epoch threshold tuning (default), `2` = softmax

## Fine-tuning GastroNet-5M for a New Task

This section describes how to adapt GastroNet-5M pretrained weights for your own endoscopy classification task.

### 1. Download GastroNet weights

Download the DINOv1-pretrained weights from [GastroNet-5M](https://cortex.thetavision.nl/dataset-provider/listing/2/) and place them in `data/assets/pretrained/`:

```bash
mkdir -p data/assets/pretrained
# Place downloaded .pth files here:
#   - RN50_GastroNet-5M_DINOv1.pth   (ResNet-50)
#   - VITS_GastroNet-5M_DINOv1.pth   (ViT-Small/16)
```

### 2. Prepare your dataset

Organize images into class folders and create a manifest CSV:

```
data/your_task/processed/
├── ClassA/
│   └── patient_001/
│       └── segment_1/
│           ├── frame_0000.jpg
│           └── frame_0001.jpg
└── ClassB/
    └── patient_002/
        └── segment_1/
            └── frame_0000.jpg
```

Create `data/your_task/splits/dataset_manifest.csv`:

```csv
relative_path,video_id,patient_id,class_name,label,segment_id,clip_key,split
ClassA/patient_001/segment_1/frame_0000.jpg,patient_001,patient_001,ClassA,0,segment_1,patient_001__segment_1,train
ClassB/patient_002/segment_1/frame_0000.jpg,patient_002,patient_002,ClassB,1,segment_1,patient_002__segment_1,train
```

Required columns: `relative_path`, `label`, `video_id`, `patient_id`, `segment_id`, `clip_key`, `split`

### 3. Create a config file

Create `configs/your_task.yaml`:

```yaml
model:
  model: vits16_gastronet    # or resnet50_gastronet
  num_classes: 1             # 1 for binary, N for N-class
  freeze_layers: 0           # 0=full fine-tune, -1=freeze backbone
  dropout_rate: 0.5

training:
  batch_size: 64
  epochs: 100
  learning_rate: 1.0e-6      # use low LR for pretrained models
  optimizer: AdamW
  weight_decay: 1.0e-2
  es_patience: 10
  equalize: true             # CLAHE contrast enhancement
```

### 4. Train

```bash
# Single train/val/test split
python -m scripts.ulcer.train \
    --config configs/your_task.yaml \
    --data-dir data/your_task/processed \
    --manifest data/your_task/splits/dataset_manifest.csv \
    --mode split

# 5-fold cross-validation
python -m scripts.ulcer.train \
    --config configs/your_task.yaml \
    --data-dir data/your_task/processed \
    --manifest data/your_task/splits/dataset_manifest.csv \
    --mode cv \
    --n-splits 5
```

### 5. Evaluate

Checkpoints are saved to `output/ulcer/models/detection/{model}/{timestamp}/best.pt`.
Metrics are logged to MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Tips

- **Learning rate**: GastroNet models benefit from lower learning rates (1e-6) compared to ImageNet-pretrained models (1e-4)
- **Freezing**: For small datasets (<1000 images), try `freeze_layers: -1` to freeze the backbone
- **Class imbalance**: The pipeline auto-computes class weights from training data
- **Clip-level aggregation**: Set `aggregate_by_clip: true` in config to aggregate frame predictions per video segment

## MLflow Tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

| Experiment | MLflow name |
|------------|-------------|
| Ulcer detection CV | `ulcer_detection` |
| Informative filtering | (logged inline) |

Checkpoints are saved to `output/ulcer/models/detection/{model}/{timestamp}/best.pt`.

## Tests

```bash
pytest
pytest tests/ --cov=src --cov=scripts
```

## Labels

| Task | Label | Meaning |
|------|-------|---------|
| Informative filtering | 1 | Informative frame |
| Informative filtering | 0 | Non-informative frame |
| Ulcer detection | 1 | Ulcer |
| Ulcer detection | 0 | Non-ulcer |

## Citation

If you use this code, please cite the paper (citation details to be added upon publication).
