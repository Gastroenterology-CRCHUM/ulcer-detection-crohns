"""
scripts/noninformative/train_noninformative.py
===============================================
Training script for the Non-Informative frame classifier.
Equivalent to notebooks/noninformative_detection.ipynb.

Pipeline
--------
    1. Load manifest from data/splits_inf/
    2. Extract features (hand-crafted + Inception-v3 bottleneck) — cached
    3. Train Random Forest
    4. Tune decision threshold on validation set
    5. Evaluate on test set with bootstrap CIs
    6. Save model + metrics + plots

Usage
-----
    # Full run (all features, bottleneck ON)
    python -m scripts.noninformative.train_noninformative

    # Hand-crafted only, specific groups
    python -m scripts.noninformative.train_noninformative --no-bottleneck --groups blur glcm edge

    # Force feature re-extraction
    python -m scripts.noninformative.train_noninformative --recompute

    # Skip feature extraction (use cache) and skip training (load existing model)
    python -m scripts.noninformative.train_noninformative --load-model output/informative/models/rf_pipeline.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import get_default_paths
from src.noninformative.features import (
    ALL_GROUPS,
    FEATURE_NAMES,
    BottleneckExtractor,
    extract_all,
    get_feature_names,
    parse_groups_arg,
)
from src.noninformative.model import NonInformativeClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _manifest_hash(manifest: pd.DataFrame) -> str:
    key_cols = ["split", "relative_path", "label"]
    rows = manifest[key_cols].sort_values(key_cols).to_csv(index=False).encode()
    return hashlib.sha256(rows).hexdigest()


def extract_or_load_features(
    manifest: pd.DataFrame,
    cache_path: Path,
    groups: list[str] | None,
    use_bottleneck: bool,
    n_jobs: int,
    batch_size: int,
    num_workers: int,
    recompute: bool,
    verbose: bool = True,
) -> dict:
    """Extract features for all splits or reload from cache."""
    current_hash = _manifest_hash(manifest)
    if cache_path.exists() and not recompute:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

        cached_hash = cached.get("manifest_hash")
        cached_groups = cached.get("groups")
        cached_bottleneck = cached.get("use_bottleneck")

        if cached_hash != current_hash:
            print("  [cache] Manifest changed — re-extracting.")
        elif cached_groups != groups:
            print(f"  [cache] Groups changed ({cached_groups} → {groups}) — re-extracting.")
        elif cached_bottleneck != use_bottleneck:
            print(
                f"  [cache] Bottleneck changed ({cached_bottleneck} → {use_bottleneck}) — re-extracting."
            )
        else:
            print("  [cache] Manifest + config unchanged — loading from cache.")
            return cached

    print("Extracting features…")
    extractor = BottleneckExtractor() if use_bottleneck else None
    result = {"manifest_hash": current_hash}

    for split_name in ("train", "val", "test"):
        df = manifest[manifest["split"] == split_name].reset_index(drop=True)
        print(f"  {split_name} ({len(df)} frames)…")
        rel_paths = df["relative_path"].tolist()
        paths = df["image_path"].tolist()
        X = extract_all(
            paths,
            groups=groups,
            use_bottleneck=use_bottleneck,
            bottleneck_extractor=extractor,
            n_jobs=n_jobs,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )
        result[f"X_{split_name}"] = X
        result[f"y_{split_name}"] = df["label"].values
        result[f"paths_{split_name}"] = rel_paths

    hc_names = get_feature_names(groups)
    bn_names = [f"bn_{i}" for i in range(2048)] if use_bottleneck else []
    result["feature_names"] = hc_names + bn_names
    result["groups"] = groups
    result["use_bottleneck"] = use_bottleneck

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Features cached → {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_feature_importances(clf, models_dir: Path, top_n: int = 20):
    if clf.feature_importances is None:
        return

    hc_imp = clf.feature_importances[clf.feature_importances.index.isin(FEATURE_NAMES)].head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    cmap = plt.colormaps["plasma"]
    _bars = ax.barh(
        hc_imp.index[::-1],
        hc_imp.values[::-1],
        color=[cmap(i / top_n) for i in range(top_n)],
        edgecolor="#333355",
        linewidth=0.6,
    )

    ax.set_xlabel("Mean decrease in impurity", color="#aaaacc")
    ax.set_title(f"Top-{top_n} hand-crafted feature importances", color="white", pad=10)
    ax.tick_params(colors="#aaaacc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout()
    out = models_dir / "feature_importances.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_evaluation(results: dict, models_dir: Path):
    cm = results["confusion_matrix"]
    labels = ["Informative", "Non-Inf."]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    axes[0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(labels, color="#aaaacc")
    axes[0].set_yticklabels(labels, color="#aaaacc")
    axes[0].set_xlabel("Predicted", color="white")
    axes[0].set_ylabel("True", color="white")
    axes[0].set_title("Confusion Matrix (normalised)", color="white")
    for i in range(2):
        for j in range(2):
            axes[0].text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] < 0.6 else "#111",
            )

    metrics = {
        "F1": results["f1"],
        "AUROC": results["roc_auc"],
        "Accuracy": results["accuracy"],
        "Sensitivity": results["sensitivity"],
        "Specificity": results["specificity"],
    }
    cmap = plt.colormaps["cool"]
    colors = [cmap(i / len(metrics)) for i in range(len(metrics))]
    bars = axes[1].bar(metrics.keys(), metrics.values(), color=colors, edgecolor="#333355")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Test metrics", color="white")
    axes[1].tick_params(colors="#aaaacc")
    for bar, val in zip(bars, metrics.values()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            color="white",
            fontsize=9,
        )
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout()
    out = models_dir / "test_evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    paths = get_default_paths()
    inf_cfg = paths.get_informative_config()
    inf_out_cfg = paths.get_task_output_config("informative")

    parser = argparse.ArgumentParser(
        description="Train non-informative frame classifier (Random Forest)"
    )
    parser.add_argument("--splits-dir", default=str(inf_cfg["splits_dir"]))
    parser.add_argument("--models-dir", default=str(inf_out_cfg["models_dir"]))
    parser.add_argument(
        "--load-model", default=None, help="Path to existing model pkl — skip training"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help=f"Hand-crafted feature groups to extract. Valid groups: {ALL_GROUPS}",
    )
    parser.add_argument(
        "--no-bottleneck", action="store_true", help="Use hand-crafted features only"
    )
    parser.add_argument(
        "--recompute", action="store_true", help="Force feature re-extraction (ignore cache)"
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument(
        "--threshold-metric", type=str, default="f1", choices=["f1", "balanced_accuracy"]
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    splits_dir = Path(args.splits_dir)
    use_bottleneck = not args.no_bottleneck
    models_dir.mkdir(parents=True, exist_ok=True)

    cache_path = models_dir / "features_cache.pkl"
    model_path = models_dir / "rf_pipeline.pkl"

    # ── 1. Manifest ───────────────────────────────────────────────────
    print("=" * 60)
    manifest_path = splits_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run scripts/noninformative/preprocess_inf.py first."
        )
    manifest = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(manifest):,} frames")
    for split in ("train", "val", "test"):
        n = (manifest["split"] == split).sum()
        print(f"  {split:<6}: {n:,}")

    # ── 2. Features ───────────────────────────────────────────────────
    print()
    groups = parse_groups_arg(args.groups)
    if groups:
        print(f"  Feature groups : {groups}")
    else:
        print(f"  Feature groups : all ({ALL_GROUPS})")
    features = extract_or_load_features(
        manifest=manifest,
        cache_path=cache_path,
        groups=groups,
        use_bottleneck=use_bottleneck,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        recompute=args.recompute,
    )
    X_train, y_train = features["X_train"], features["y_train"]
    X_val, y_val = features["X_val"], features["y_val"]
    X_test, y_test = features["X_test"], features["y_test"]
    feat_names = features["feature_names"]
    print(f"Feature matrix: {X_train.shape[1]} features")

    # ── 3. Train or load ──────────────────────────────────────────────
    print()
    if args.load_model:
        clf = NonInformativeClassifier.load(args.load_model)
    else:
        clf = NonInformativeClassifier(
            rf_params={
                "n_estimators": args.n_estimators,
                "max_features": args.max_features,
                "min_samples_leaf": args.min_samples_leaf,
                "class_weight": "balanced",
                "random_state": args.seed,
                "n_jobs": args.n_jobs,
            }
        )
        clf.fit(X_train, y_train, feature_names=feat_names)

        # ── 4. Threshold tuning ───────────────────────────────────────
        print()
        clf.tune_threshold(X_val, y_val, metric=args.threshold_metric)
        clf.save(model_path)

    # ── 5. Evaluation ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    results = clf.evaluate(X_test, y_test, n_bootstrap=args.n_bootstrap)

    test_paths = features["paths_test"]
    test_df = (
        manifest[manifest["relative_path"].isin(test_paths)]
        .set_index("relative_path")
        .loc[test_paths]
        .reset_index()
    )
    if len(test_df) != len(test_paths):
        raise RuntimeError(
            f"{len(test_paths) - len(test_df)} frame(s) du cache absentes du manifest. "
            "Lance avec --recompute."
        )
    test_df["pred_label"] = results["predictions"]
    test_df["pred_prob"] = results["probabilities"].round(4)
    test_df["pred_name"] = test_df["pred_label"].map({1: "Informative", 0: "Non-Informative"})
    test_df["correct"] = (test_df["pred_label"] == test_df["label"]).astype(int)

    def _tag(t, p):
        if t == 1 and p == 1:
            return "TP"
        if t == 0 and p == 1:
            return "FP"
        if t == 0 and p == 0:
            return "TN"
        return "FN"

    test_df["outcome"] = [_tag(t, p) for t, p in zip(test_df["label"], test_df["pred_label"])]

    pred_path = models_dir / "test_predictions.csv"
    test_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved → {pred_path}")

    # ── 6. Metrics JSON ───────────────────────────────────────────────
    metrics = {
        "threshold": clf.threshold,
        "f1": results["f1"],
        "f1_ci": results["f1_ci"],
        "roc_auc": results["roc_auc"],
        "roc_auc_ci": results["roc_auc_ci"],
        "accuracy": results["accuracy"],
        "sensitivity": results["sensitivity"],
        "specificity": results["specificity"],
        "groups": args.groups or "all",
        "bottleneck": use_bottleneck,
        "n_features": int(X_train.shape[1]),
    }
    with open(models_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # ── 7. Plots ──────────────────────────────────────────────────────
    print()
    print("Saving plots…")
    plot_feature_importances(clf, models_dir)
    plot_evaluation(results, models_dir)

    print()
    print("=" * 60)
    print(f"Done. Artefacts in {models_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
