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
│   ├── noninformative/                    Informative-frame RF classifier (train, filter, review)
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

### Informative-frame classifier (standalone)

The RF classifier used in preprocessing can be retrained independently:

```bash
python -m scripts.noninformative.preprocess_inf        # build training manifest
python -m scripts.noninformative.train_noninformative   # train RF classifier
```

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
