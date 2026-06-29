"""
src/noninformative/features.py
==============================
Feature extraction for non-informative frame classification.

Two feature families (following the paper):

1. Hand-crafted features (43 total) in HSV colour space:
   - Specular reflection mask         (1)
   - Intensity statistics H/S/V       (12 = 4 stats × 3 channels)
   - Edge features                    (2)
   - GLCM statistics H/S/V            (12 = 4 stats × 3 channels)
   - Blur measures                    (10)
   - Bubble detection                 (6)

2. Bottleneck features (2048):
   - Inception-v3 global average pooling activations

References
----------
    [Canny]  Canny (1986)
    [GLCM]   Haralick et al. (1973)
    [LAPE]   Subbarao et al. focus measure (Laplacian energy)
    [DCTR]   Shen & Chen focus measure (DCT ratio)
"""

from __future__ import annotations

import atexit
import os
import platform
import re
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional — only needed for bottleneck features
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import Inception_V3_Weights, inception_v3

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "reflection": [
        "reflection_ratio",
    ],
    "intensity": [
        f"{stat}_{ch}" for ch in ("H", "S", "V") for stat in ("mean", "var", "skew", "kurt")
    ],
    "edge": [
        "edge_ratio",
        "n_lines",
    ],
    "glcm": [
        f"glcm_{stat}_{ch}"
        for ch in ("H", "S", "V")
        for stat in ("contrast", "energy", "homogeneity", "correlation")
    ],
    "blur": [
        "blur_mean_diff",
        "blur_std_diff",
        "blur_freq",
        "lape_H",
        "lape_S",
        "lape_V",
        "dctr_H",
        "dctr_S",
        "dctr_V",
        "blur_combined",
    ],
    "bubbles": [
        "bubble_circle_count",
        "bubble_contour_ratio",
        "bubble_highlight_ratio",
        "bubble_local_var",
        "bubble_hue_entropy",
        "bubble_score",
    ],
}

ALL_GROUPS: list[str] = ["reflection", "intensity", "edge", "glcm", "blur", "bubbles"]

FEATURE_NAMES: list[str] = [name for g in ALL_GROUPS for name in FEATURE_GROUPS[g]]

assert len(FEATURE_NAMES) == 43, f"Expected 43 features, got {len(FEATURE_NAMES)}"

GLCM_LEVELS = 32
GLCM_DIST = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
DCT_BLOCK = 15

# Hough circle parameters — tuned for bubble sizes in colonoscopy (10–80 px radius)
_BUBBLE_MIN_RADIUS = 8
_BUBBLE_MAX_RADIUS = 80
_BUBBLE_DP = 1.2  # inverse accumulator resolution
_BUBBLE_MIN_DIST = 15  # minimum distance between circle centres

# Highlight detection: a bubble highlight is a small, very bright blob
_HIGHLIGHT_MIN_BRIGHTNESS = 220  # out of 255 (V channel)
_HIGHLIGHT_MAX_AREA_FRAC = 0.002  # each blob ≤ 0.2 % of ROI area → filters large reflections

# Local variance window
_LOCAL_VAR_WINDOW = 9


def get_feature_names(groups: list[str] | None = None) -> list[str]:
    """
    Return the ordered list of feature names for the given groups.

    Args:
        groups : Subset of ALL_GROUPS to include.
                 None or empty → all 43 features.
    """
    active = groups if groups else ALL_GROUPS
    unknown = set(active) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}. Valid: {ALL_GROUPS}")
    return [name for g in ALL_GROUPS if g in active for name in FEATURE_GROUPS[g]]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255)
        return image.astype(np.uint8)
    return image


def build_reflection_mask(hsv: np.ndarray, sat_threshold: int = 30) -> np.ndarray:
    S = hsv[:, :, 1]
    _, mask = cv2.threshold(S, sat_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def build_roi_mask(image: np.ndarray, threshold: int = 30) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def inpaint_reflection_and_background(
    channel: np.ndarray,
    reflection_mask: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    combined = cv2.bitwise_or(reflection_mask, cv2.bitwise_not(roi_mask))
    inpainted = cv2.inpaint(channel, combined, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted


# ---------------------------------------------------------------------------
# Individual feature groups
# ---------------------------------------------------------------------------


def _reflection_feature(reflection_mask, roi_mask) -> list[float]:
    roi_area = (roi_mask > 0).sum()
    if roi_area == 0:
        return [0.0]
    ref_in_roi = ((reflection_mask > 0) & (roi_mask > 0)).sum()
    return [float(ref_in_roi) / roi_area]


def _intensity_statistics(hsv, reflection_mask, roi_mask) -> list[float]:
    valid_mask = (roi_mask > 0) & (reflection_mask == 0)
    feats: list[float] = []
    for c in range(3):
        pixels = hsv[:, :, c][valid_mask].astype(float)
        if len(pixels) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend(
                [
                    float(pixels.mean()),
                    float(pixels.var()),
                    float(skew(pixels)),
                    float(kurtosis(pixels)),
                ]
            )
    return feats


def _edge_features(hsv, reflection_mask, roi_mask) -> list[float]:
    V = hsv[:, :, 2]
    V_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(V)
    edges = cv2.Canny(V_clahe, 50, 150)
    edges[reflection_mask > 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
    for lbl in range(1, num_labels):
        pts = np.argwhere(labels == lbl)
        if len(pts) < 5:
            continue
        try:
            (_, _), (ma, mi), _ = cv2.fitEllipse(pts[:, ::-1].astype(np.float32))
            ma, mi = max(ma, mi), min(ma, mi)
            if ma > 0:
                ratio = np.clip(mi / ma, 0.0, 1.0)
                ecc = np.sqrt(1.0 - ratio**2)
                if ecc < 0.9:
                    edges[labels == lbl] = 0
        except cv2.error:
            pass

    roi_area = max(1, (roi_mask > 0).sum())
    edge_ratio = float(edges.sum() // 255) / roi_area
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    n_lines = float(len(lines)) if lines is not None else 0.0
    return [edge_ratio, n_lines]


def _glcm_features(hsv, reflection_mask, roi_mask) -> list[float]:
    feats: list[float] = []
    for c in range(3):
        ch = inpaint_reflection_and_background(hsv[:, :, c], reflection_mask, roi_mask)
        ch_q = (ch.astype(float) / 256 * GLCM_LEVELS).clip(0, GLCM_LEVELS - 1).astype(np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glcm = graycomatrix(
                ch_q,
                distances=GLCM_DIST,
                angles=GLCM_ANGLES,
                levels=GLCM_LEVELS,
                symmetric=True,
                normed=True,
            )
        for prop in ("contrast", "energy", "homogeneity", "correlation"):
            feats.append(float(graycoprops(glcm, prop).mean()))
    return feats


def _lape(channel: np.ndarray) -> float:
    lap = cv2.Laplacian(channel.astype(np.float64), cv2.CV_64F)
    return float((lap**2).sum())


def _dctr(channel: np.ndarray, block_size: int = DCT_BLOCK) -> float:
    H, W = channel.shape
    total = 0.0
    count = 0
    ch = channel.astype(np.float32)
    for u in range(0, H - block_size, block_size):
        for v in range(0, W - block_size, block_size):
            block = ch[u : u + block_size, v : v + block_size]
            dct = cv2.dct(block)
            dc = dct[0, 0] ** 2
            ac = (dct**2).sum() - dc
            if dc > 0:
                total += ac / dc
                count += 1
    return total / max(1, count)


def _blur_features(hsv, reflection_mask, roi_mask) -> list[float]:
    V = inpaint_reflection_and_background(hsv[:, :, 2], reflection_mask, roi_mask)
    V_blur = cv2.GaussianBlur(V, (15, 15), 0)
    diff = np.abs(V.astype(float) - V_blur.astype(float))
    blur_mean = float(diff.mean())
    blur_std = float(diff.std())
    blur_freq = float(cv2.Laplacian(V, cv2.CV_64F).var())

    channels_inp = [
        inpaint_reflection_and_background(hsv[:, :, c], reflection_mask, roi_mask) for c in range(3)
    ]
    lapes = [_lape(ch) for ch in channels_inp]
    dctrs = [_dctr(ch) for ch in channels_inp]
    blur_combined = float(np.mean(lapes))
    return [blur_mean, blur_std, blur_freq] + lapes + dctrs + [blur_combined]


def _bubble_features(
    image_rgb: np.ndarray,
    hsv: np.ndarray,
    reflection_mask: np.ndarray,
    roi_mask: np.ndarray,
) -> list[float]:
    """
    Bubble-specific features (6).

    Why not use edge_ratio?
    -----------------------
    ``_edge_features`` removes near-circular edge components (eccentricity < 0.9)
    on purpose — so edge_ratio suppresses bubble edges rather than counting them.
    These features operate on the **original** circular structures instead.

    Features
    --------
    bubble_circle_count   : Number of Hough circles detected in the V channel.
                            Bubbles produce clear circular structures; mucosal
                            folds and blur do not.
    bubble_contour_ratio  : Fraction of external contours with circularity > 0.75.
                            Circularity = 4π·area / perimeter².
    bubble_highlight_ratio: Fraction of ROI covered by small, very bright blobs.
                            Each bubble has a tiny specular highlight at its apex.
                            Large reflections are excluded by the area cap.
    bubble_local_var      : Mean local variance in a 9×9 sliding window on V.
                            Bubble clusters produce many sharp local transitions.
    bubble_hue_entropy    : Shannon entropy of the H-channel histogram.
                            Bubble membranes refract light, adding hue diversity.
    bubble_score          : Weighted combination (0–1 clamped) used as a single
                            summary feature for the classifier.
    """
    V = hsv[:, :, 2]
    roi_area = max(1, (roi_mask > 0).sum())

    # ── 1. Hough circles ───────────────────────────────────────────────────
    V_blur = cv2.GaussianBlur(V, (9, 9), 2)
    circles = cv2.HoughCircles(
        V_blur,
        cv2.HOUGH_GRADIENT,
        dp=_BUBBLE_DP,
        minDist=_BUBBLE_MIN_DIST,
        param1=60,  # Canny upper threshold inside HoughCircles
        param2=25,  # accumulator threshold — lower = more circles detected
        minRadius=_BUBBLE_MIN_RADIUS,
        maxRadius=_BUBBLE_MAX_RADIUS,
    )
    n_circles = float(len(circles[0])) if circles is not None else 0.0

    # Normalise: cap at 50 circles (a frame packed with bubbles)
    bubble_circle_count = min(n_circles, 50.0) / 50.0

    # ── 2. Contour circularity ─────────────────────────────────────────────
    # Threshold on V to get candidate blobs, then measure circularity
    _, thresh = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, roi_mask)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circular_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:  # ignore tiny noise blobs
            continue
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity > 0.75:
            circular_count += 1

    total_contours = max(1, len([c for c in contours if cv2.contourArea(c) >= 20]))
    bubble_contour_ratio = float(circular_count) / total_contours

    # ── 3. Highlight ratio ─────────────────────────────────────────────────
    # Each bubble has a tiny bright specular spot at its apex.
    # Large highlights (> 0.2 % of ROI) are large reflections → excluded.
    _, bright_mask = cv2.threshold(V, _HIGHLIGHT_MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.bitwise_and(bright_mask, roi_mask)
    bright_mask = cv2.bitwise_and(
        bright_mask, cv2.bitwise_not(reflection_mask)
    )  # exclude large reflections

    n_bright, bright_labels, bright_stats, _ = cv2.connectedComponentsWithStats(bright_mask)
    max_blob_area = _HIGHLIGHT_MAX_AREA_FRAC * roi_area
    small_highlight_pixels = 0
    for lbl in range(1, n_bright):
        blob_area = bright_stats[lbl, cv2.CC_STAT_AREA]
        if blob_area <= max_blob_area:
            small_highlight_pixels += blob_area

    bubble_highlight_ratio = float(small_highlight_pixels) / roi_area

    # ── 4. Local variance ──────────────────────────────────────────────────
    V_f = V.astype(np.float32)
    V_sq = V_f**2
    kernel = np.ones((_LOCAL_VAR_WINDOW, _LOCAL_VAR_WINDOW), dtype=np.float32) / (
        _LOCAL_VAR_WINDOW**2
    )
    mean_sq = cv2.filter2D(V_f, -1, kernel) ** 2
    sq_mean = cv2.filter2D(V_sq, -1, kernel)
    local_var = np.maximum(sq_mean - mean_sq, 0.0)
    bubble_local_var = float(local_var[roi_mask > 0].mean()) / (255.0**2)

    # ── 5. Hue entropy ─────────────────────────────────────────────────────
    H_channel = hsv[:, :, 0]
    H_in_roi = H_channel[roi_mask > 0]
    if len(H_in_roi) > 0:
        hist, _ = np.histogram(H_in_roi, bins=36, range=(0, 180))
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-9
        entropy = -float(np.sum(hist * np.log2(hist + 1e-9)))
        bubble_hue_entropy = entropy / np.log2(36)  # normalise to [0, 1]
    else:
        bubble_hue_entropy = 0.0

    # ── 6. Composite bubble score ──────────────────────────────────────────
    # Weights chosen to reflect discriminative power:
    #   Hough circles and contour circularity are the most direct indicators.
    #   Highlights and hue entropy are supporting evidence.
    #   Local variance alone is too noisy (shared with blur and proximity).
    bubble_score = float(
        np.clip(
            0.35 * bubble_circle_count
            + 0.25 * bubble_contour_ratio
            + 0.20 * min(bubble_highlight_ratio / 0.05, 1.0)  # cap at 5 % coverage
            + 0.10 * bubble_local_var / 0.10  # cap at 10 % normalised var
            + 0.10 * bubble_hue_entropy,
            0.0,
            1.0,
        )
    )

    return [
        bubble_circle_count,
        bubble_contour_ratio,
        bubble_highlight_ratio,
        bubble_local_var,
        bubble_hue_entropy,
        bubble_score,
    ]


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_handcrafted(
    image_rgb: np.ndarray,
    groups: list[str] | None = None,
) -> np.ndarray:
    """
    Extract hand-crafted features from an RGB image.

    Args:
        image_rgb : H×W×3 uint8 RGB array.
        groups    : Feature groups to compute. None = all 6 groups (43 features).

    Returns:
        1-D float64 array of shape (D,).
    """
    active = set(groups) if groups else set(ALL_GROUPS)
    unknown = active - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"Unknown group(s): {unknown}. Valid: {ALL_GROUPS}")

    img = _to_uint8(image_rgb)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    roi = build_roi_mask(img)
    refl = build_reflection_mask(hsv)

    _group_fns = {
        "reflection": lambda: _reflection_feature(refl, roi),
        "intensity": lambda: _intensity_statistics(hsv, refl, roi),
        "edge": lambda: _edge_features(hsv, refl, roi),
        "glcm": lambda: _glcm_features(hsv, refl, roi),
        "blur": lambda: _blur_features(hsv, refl, roi),
        "bubbles": lambda: _bubble_features(img, hsv, refl, roi),
    }
    feats: list[float] = []
    for g in ALL_GROUPS:
        if g in active:
            feats.extend(_group_fns[g]())

    return np.array(feats, dtype=np.float64)


def _load_and_extract(item, groups):
    """Load image from path (or use array) then extract features. Module-level for joblib."""
    if isinstance(item, (str, Path)):
        img = cv2.imread(str(item))
        if img is None:
            raise FileNotFoundError(f"Cannot load: {item}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = item
    return extract_handcrafted(img, groups)


def extract_handcrafted_batch(
    images: list[np.ndarray] | list[str] | list[Path],
    groups: list[str] | None = None,
    verbose: bool = True,
    n_jobs: int = -1,
    from_paths: bool = False,
) -> np.ndarray:
    """Extract hand-crafted features for a list of images in parallel (CPU)."""
    if n_jobs == 1:
        iterator = tqdm(images, desc="Hand-crafted features") if verbose else images
        return np.vstack([_load_and_extract(item, groups) for item in iterator])

    n_workers = min(os.cpu_count(), 8) if n_jobs == -1 else n_jobs
    n_feats = len(get_feature_names(groups))
    if verbose:
        active_groups = groups or ALL_GROUPS
        print(
            f"  Hand-crafted ({', '.join(active_groups)}): "
            f"{len(images)} images × {n_feats} features × {n_workers} CPU workers"
        )

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_load_and_extract)(item, groups)
        for item in (tqdm(images, desc="Hand-crafted (parallel)") if verbose else images)
    )
    return np.vstack(results)


# ---------------------------------------------------------------------------
# CLI helper — parse --groups robustly
# ---------------------------------------------------------------------------


def parse_groups_arg(raw: list[str] | None) -> list[str] | None:
    """
    Normalise the ``--groups`` CLI argument regardless of quoting style.

    Handles all three common invocations on Windows PowerShell / bash:

        --groups blur glcm edge              → ["blur", "glcm", "edge"]
        --groups '["blur", "glcm"]'          → ["blur", "glcm"]
        --groups "[\"blur\", \"glcm\"]"      → ["blur", "glcm"]
        --groups blur,glcm,edge              → ["blur", "glcm", "edge"]

    Returns None if raw is None or empty (→ all groups).
    """
    if not raw:
        return None

    # Join in case argparse split a JSON string across multiple tokens
    joined = " ".join(raw).strip()

    # Strip surrounding brackets and quotes if the user passed a JSON array
    if joined.startswith("["):
        tokens = re.findall(r"[a-zA-Z_]+", joined)
    elif "," in joined:
        # comma-separated without brackets: "blur,glcm,edge"
        tokens = [t.strip() for t in joined.split(",")]
    else:
        # already space-separated tokens: blur glcm edge
        tokens = joined.split()

    # Validate
    unknown = set(tokens) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}. Valid: {ALL_GROUPS}")
    return tokens or None


# ---------------------------------------------------------------------------
# Module-level dataset (must be at top-level scope for DataLoader pickling)
# ---------------------------------------------------------------------------


class _ImgDataset:
    def __init__(self, items, transform):
        self._transform = transform
        if len(items) == 0:
            self._paths = []
            self._arrays = None
        elif isinstance(items[0], (str, Path)):
            self._paths = [str(p) for p in items]
            self._arrays = None
        else:
            self._paths = None
            arr = np.stack([_to_uint8(img) for img in items])
            tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
            self._mmap_path = tmp.name
            tmp.close()
            np.save(self._mmap_path, arr)
            atexit.register(lambda p=self._mmap_path: Path(p).unlink(missing_ok=True))
            self._arrays = None
            self._mmap_shape = arr.shape
            self._mmap_dtype = arr.dtype

    def __len__(self):
        return len(self._paths) if self._paths is not None else self._mmap_shape[0]

    def __getitem__(self, idx):
        if self._paths is not None:
            img = cv2.imread(self._paths[idx])
            if img is None:
                raise FileNotFoundError(self._paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            if self._arrays is None:
                self._arrays = np.load(self._mmap_path, mmap_mode="r")
            img = self._arrays[idx]
        return self._transform(_to_uint8(img))


# ---------------------------------------------------------------------------
# Inception-v3 bottleneck features (2048-d)
# ---------------------------------------------------------------------------


class BottleneckExtractor:
    def __init__(self, device: str | None = None):
        if not _TORCH_AVAILABLE:
            raise ImportError("pip install torch torchvision")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._hook_output = None
        self._transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(299),
                T.CenterCrop(299),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load(self):
        if self._model is not None:
            return
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.eval()
        model.to(self.device)

        def _hook(module, inp, out):
            self._hook_output = out

        model.avgpool.register_forward_hook(_hook)
        self._model = model

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load()
        tensor = self._transform(_to_uint8(image_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self._model(tensor)
        return self._hook_output.squeeze().cpu().numpy().astype(np.float32)

    def extract_batch(
        self,
        images: list[np.ndarray] | list[str] | list[Path],
        batch_size: int = 128,
        num_workers: int = 4,
        verbose: bool = True,
    ) -> np.ndarray:
        self._load()
        is_paths = len(images) > 0 and isinstance(images[0], (str, Path))
        effective_workers = num_workers
        if not is_paths and platform.system() == "Windows" and num_workers > 0:
            if verbose:
                print("  [Windows] Forcing num_workers=0 for array inputs.")
            effective_workers = 0

        dataset = _ImgDataset(images, self._transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=effective_workers,
            pin_memory=(self.device != "cpu"),
            prefetch_factor=2 if effective_workers > 0 else None,
            persistent_workers=(effective_workers > 0),
        )
        if verbose:
            print(
                f"  Bottleneck: {len(images)} images | "
                f"batch={batch_size} workers={effective_workers} device={self.device}"
            )
        results = []
        it = tqdm(loader, desc="Bottleneck") if verbose else loader
        for batch_tensors in it:
            batch_tensors = batch_tensors.to(self.device, non_blocking=True)
            with torch.no_grad():
                self._model(batch_tensors)
            feats = self._hook_output.squeeze(dim=-1).squeeze(dim=-1).cpu().numpy()
            results.append(feats)
        return np.vstack(results).astype(np.float32)


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------


def extract_all(
    images: list[np.ndarray] | list[str] | list[Path],
    groups: list[str] | None = None,
    use_bottleneck: bool = True,
    bottleneck_extractor: BottleneckExtractor | None = None,
    n_jobs: int = -1,
    batch_size: int = 128,
    num_workers: int = 4,
    verbose: bool = True,
) -> np.ndarray:
    """Extract and concatenate hand-crafted + bottleneck (2048) features."""
    is_paths = len(images) > 0 and isinstance(images[0], (str, Path))
    hc = extract_handcrafted_batch(
        images,
        groups=groups,
        verbose=verbose,
        n_jobs=n_jobs,
        from_paths=is_paths,
    )
    if not use_bottleneck:
        return hc
    extractor = bottleneck_extractor or BottleneckExtractor()
    bn = extractor.extract_batch(
        images, batch_size=batch_size, num_workers=num_workers, verbose=verbose
    )
    return np.hstack([hc, bn])
