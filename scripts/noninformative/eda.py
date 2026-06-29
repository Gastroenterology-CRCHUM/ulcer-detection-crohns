"""EDA for the informative / non-informative frame classification pipeline.

Reports:
- Frame counts per class (Informative / Non-Informative)
- Cause breakdown for non-informative frames
- Split-level diagnostics (label balance)
- Stratification by class × cause

Input : data/informative/splits/dataset_manifest.csv
Output: results/informative/eda/
        ├── class_distribution.png
        ├── cause_distribution.png
        ├── split_label_distribution.png
        ├── frames_per_patient_hist.png
        ├── eda_report.txt
        └── split_label_table.csv

Usage
-----
    python scripts/noninformative/eda.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import get_default_paths

plt.style.use("seaborn-v0_8-whitegrid")

SPLITS = ["train", "val", "test"]
CLASS_NAMES = {1: "Informative", 0: "Non-Informative"}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["label"].map(CLASS_NAMES).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(v), ha="center")
    ax.set_title("Frames per class")
    ax.set_ylabel("Frames")
    _save(fig, output_dir / "class_distribution.png")


def plot_cause_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "cause" not in df.columns:
        return
    non_inf = df[df["label"] == 0]
    if non_inf.empty:
        return
    counts = non_inf["cause"].fillna("Unknown").value_counts()
    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 0.8), 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Non-Informative frames by cause")
    ax.set_ylabel("Frames")
    plt.xticks(rotation=30, ha="right")
    _save(fig, output_dir / "cause_distribution.png")


def plot_split_label_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "split" not in df.columns:
        return
    ct = pd.crosstab(df["split"], df["label"].map(CLASS_NAMES))
    present = [s for s in SPLITS if s in ct.index]
    ct = ct.loc[present]
    fig, ax = plt.subplots(figsize=(7, 4))
    ct.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Frames per split and class")
    ax.set_ylabel("Frames")
    ax.legend(title="Class")
    _save(fig, output_dir / "split_label_distribution.png")


def plot_frames_per_patient(df: pd.DataFrame, output_dir: Path) -> None:
    if "patient_id" not in df.columns:
        return
    fpp = df.groupby("patient_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fpp, bins=30, edgecolor="white")
    ax.axvline(fpp.mean(), color="red", linestyle="--", label=f"Mean = {fpp.mean():.0f}")
    ax.set_title("Frames per patient")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Patients")
    ax.legend()
    _save(fig, output_dir / "frames_per_patient_hist.png")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def generate_report(df: pd.DataFrame) -> str:
    W = 76
    lines: list[str] = []

    def h1(t: str) -> None:
        lines.extend(["=" * W, t.center(W), "=" * W])

    def h2(t: str) -> None:
        lines.extend([f"\n{t}", "-" * len(t)])

    h1("INFORMATIVE PIPELINE - EXPLORATORY DATA ANALYSIS REPORT")

    if df.empty:
        lines.append("\n  No manifest data available.")
        return "\n".join(lines)

    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else "N/A"

    # ── 1. Overview ───────────────────────────────────────────────────────
    h2("1. DATASET OVERVIEW")
    col_w = 16
    lines.append(f"  {'':30} {'Total':>{col_w}} {'Informative':>{col_w}} {'Non-Inf.':>{col_w}}")
    lines.append("  " + "-" * (30 + col_w * 3))

    def _fmt(v: int | str) -> str:
        return f"{v:>{col_w},}" if isinstance(v, int) else f"{v!s:>{col_w}}"

    def _row(label: str, total: int | str, inf: int | str, noninf: int | str) -> str:
        return f"  {label:<30} {_fmt(total)} {_fmt(inf)} {_fmt(noninf)}"

    n_inf = int((df["label"] == 1).sum())
    n_ni = int((df["label"] == 0).sum())

    lines.append(_row("Patients", int(n_patients) if isinstance(n_patients, int) else 0, "-", "-"))
    lines.append(_row("Frames", len(df), n_inf, n_ni))

    if "patient_id" in df.columns:
        fpp = df.groupby("patient_id").size()
        lines.append(
            f"\n  Frames/patient  mean {fpp.mean():.1f} ± {fpp.std():.1f}"
            f"   [min {int(fpp.min())} – max {int(fpp.max())} | median {fpp.median():.0f}]"
        )

    # ── 2. Cause breakdown ────────────────────────────────────────────────
    if "cause" in df.columns:
        h2("2. NON-INFORMATIVE CAUSE BREAKDOWN")
        ni_df = df[df["label"] == 0]
        if not ni_df.empty:
            counts = ni_df["cause"].fillna("Unknown").value_counts()
            for cause, n in counts.items():
                pct = n / len(ni_df) * 100
                lines.append(f"  {cause:<30} {n:>6,}  ({pct:.1f}%)")

    # ── 3. Split statistics ────────────────────────────────────────────────
    if "split" not in df.columns:
        return "\n".join(lines)

    present = [s for s in SPLITS if s in df["split"].values]
    h2("3. SPLIT STATISTICS")
    col_w = 12
    lines.append(f"  {'':30}" + "".join(f"{s:>{col_w}}" for s in present))
    lines.append("  " + "-" * (30 + col_w * len(present)))

    def _srow(label: str, vals: list) -> str:
        return f"  {label:<30}" + "".join(
            f"{v:>{col_w},}" if isinstance(v, int) else f"{v!s:>{col_w}}" for v in vals
        )

    if "patient_id" in df.columns:
        lines.append(
            _srow("Patients", [df[df["split"] == s]["patient_id"].nunique() for s in present])
        )
    lines.append(_srow("Frames (total)", [int((df["split"] == s).sum()) for s in present]))
    for lbl, name in CLASS_NAMES.items():
        lines.append(
            _srow(
                f"  {name}",
                [int(((df["split"] == s) & (df["label"] == lbl)).sum()) for s in present],
            )
        )
    n_total = len(df)
    pcts = [f"{(df['split'] == s).sum() / n_total * 100:.1f}%" for s in present]
    lines.append(f"  {'% of total frames':<30}" + "".join(f"{p:>{col_w}}" for p in pcts))
    lines.append("")
    lines.append(f"  {'Class split distribution':<30}" + "".join(f"{s:>{col_w}}" for s in present))
    lines.append("  " + "-" * (30 + col_w * len(present)))
    for lbl, name in CLASS_NAMES.items():
        class_total = (df["label"] == lbl).sum()
        vals = []
        for s in present:
            n = int(((df["split"] == s) & (df["label"] == lbl)).sum())
            vals.append(f"{n / class_total * 100:.1f}%" if class_total else "-")
        lines.append(f"  {name:<30}" + "".join(f"{v:>{col_w}}" for v in vals))

    # ── 4. Stratification by class × cause ───────────────────────────────
    h2("4. STRATIFICATION BY CLASS AND CAUSE (frame-level splits)")
    lines.append("  Note: splits are at the frame level — patient overlap across splits")
    lines.append("        is expected and not a data-leakage concern for this pipeline.")

    def _strat_bin(row: pd.Series) -> str:
        if row["label"] == 1:
            return "Informative"
        cause = str(row.get("cause", "")).strip()
        return f"Non-Informative / {cause}" if cause else "Non-Informative"

    df = df.copy()
    df["_strat_bin"] = df.apply(_strat_bin, axis=1)
    strat_bins = sorted(df["_strat_bin"].unique(), key=lambda x: (x != "Informative", x))

    col_w = 10
    lines.append("")
    lines.append(
        f"  {'Stratum':<38}" + "".join(f"{s:>{col_w}}" for s in present) + f"{'Total':>{col_w}}"
    )
    lines.append("  " + "-" * (38 + col_w * (len(present) + 1)))
    for bin_name in strat_bins:
        counts = [int(((df["split"] == s) & (df["_strat_bin"] == bin_name)).sum()) for s in present]
        total = sum(counts)
        lines.append(
            f"  {bin_name:<38}" + "".join(f"{c:>{col_w},}" for c in counts) + f"{total:>{col_w},}"
        )
    totals = [int((df["split"] == s).sum()) for s in present]
    lines.append(
        f"  {'Total':<38}" + "".join(f"{c:>{col_w},}" for c in totals) + f"{sum(totals):>{col_w},}"
    )

    lines.append("\n" + "=" * W)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    paths = get_default_paths()
    parser = argparse.ArgumentParser(
        description="EDA for the informative / non-informative classification pipeline."
    )
    parser.add_argument("--splits-dir", type=str, default=str(paths.informative_splits_dir))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(paths.results_informative_dir / "eda"),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = splits_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        print(
            f"Manifest not found: {manifest_path} — run scripts/noninformative/preprocess_inf.py first."
        )
        return

    df = pd.read_csv(manifest_path)

    plot_class_distribution(df, output_dir)
    plot_cause_distribution(df, output_dir)
    plot_split_label_distribution(df, output_dir)
    plot_frames_per_patient(df, output_dir)

    if {"split", "label"}.issubset(df.columns):
        pd.crosstab(df["split"], df["label"], dropna=False).to_csv(
            output_dir / "split_label_table.csv"
        )

    report = generate_report(df)
    print(report)
    (output_dir / "eda_report.txt").write_text(report, encoding="utf-8")
    print(f"\nEDA outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
