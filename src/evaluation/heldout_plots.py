"""
src/evaluation/heldout_plots.py
-------------------------------
Figures of the held-out analysis. Callers set the matplotlib backend/style.

Public API
----------
plot_forest(pair_df, ci_lo_col, ci_hi_col, title, output_path)
plot_delong_pairs(pair_df, output_path)
plot_ensemble_metrics(ensemble_df, title, output_path)
plot_pairwise_matrix(names, matrix_df, output_path)
plot_fold_dispersion_table(dispersion_df, output_path)
plot_ensemble_roc(configs, cohort, output_path)
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.heldout import METRICS, ConfigPredictions, HeldoutCohort
from src.evaluation.plot_tables import column_widths, draw_config_table, text_lengths
from src.evaluation.plots import plot_roc_curves
from src.evaluation.style import (
    STATUS_NEUTRAL,
    STATUS_NOT_SIGNIFICANT,
    STATUS_SIGNIFICANT,
    config_color,
    config_label,
    config_labels,
    metric_label,
)


def plot_forest(
    pair_df: pd.DataFrame, ci_lo_col: str, ci_hi_col: str, title: str, output_path
) -> None:
    """One row per comparison pair. Point = plug-in ensemble DeltaAUROC (A - B), whiskers =
    the bootstrap CI named by ci_lo_col/ci_hi_col. Color marks whether that CI excludes
    zero (the primary significance call). DeLong significance (ensemble/best-fold, kept
    for traceability, see module docstring) is reported separately in delong_pairs.png,
    not annotated here."""
    n = len(pair_df)
    fig, ax = plt.subplots(figsize=(9.5, max(3.5, n * 1.1)))

    labels = []
    for i, row in enumerate(pair_df.itertuples()):
        pt = row.delta_auroc_ensemble
        lo, hi = getattr(row, ci_lo_col), getattr(row, ci_hi_col)
        significant = lo > 0 or hi < 0
        color = STATUS_SIGNIFICANT if significant else STATUS_NEUTRAL
        ax.errorbar(
            pt,
            i,
            xerr=[[max(pt - lo, 0.0)], [max(hi - pt, 0.0)]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=8,
            zorder=3,
        )
        labels.append(
            f"{row.pair_id}\nA: {config_label(row.config_a)}\nB: {config_label(row.config_b)}"
        )

    ax.axvline(0, linestyle="--", color="#c3c2b7", linewidth=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("DeltaAUROC  (A - B)")
    ax.set_title(title)
    legend_elems = [
        mpatches.Patch(facecolor=STATUS_SIGNIFICANT, label="Bootstrap CI excludes 0 (significant)"),
        mpatches.Patch(facecolor=STATUS_NEUTRAL, label="Bootstrap CI includes 0 (not significant)"),
    ]
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    ax.legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=1,
        fontsize=8,
        frameon=True,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delong_pairs(pair_df: pd.DataFrame, output_path) -> None:
    """Small grouped-bar panel, one row per comparison pair: DeLong z-statistic for the
    ensemble prediction and for the best-fold prediction, side by side. Bar color marks
    Holm-Bonferroni-adjusted significance. This is the traceability check that sits
    alongside the primary bootstrap CI in effect_decomposition_bca.png, not a
    replacement for it (see module docstring)."""
    n = len(pair_df)
    fig, ax = plt.subplots(figsize=(7, max(2.5, n * 0.9)))

    y = np.arange(n)
    height = 0.32
    ens_z = pair_df["delong_ensemble_z"].to_numpy()
    bf_z = pair_df["delong_bestfold_z"].to_numpy()
    ens_sig = pair_df["delong_ensemble_p_holm"].to_numpy() < 0.05
    bf_sig = pair_df["delong_bestfold_p_holm"].to_numpy() < 0.05

    ax.barh(
        y + height / 2,
        ens_z,
        height=height,
        color=[STATUS_SIGNIFICANT if s else STATUS_NEUTRAL for s in ens_sig],
        edgecolor="#0b0b0b",
        linewidth=0.5,
    )
    ax.barh(
        y - height / 2,
        bf_z,
        height=height,
        color=[STATUS_SIGNIFICANT if s else STATUS_NEUTRAL for s in bf_sig],
        edgecolor="#0b0b0b",
        linewidth=0.5,
        hatch="//",
    )
    max_abs = max(np.abs(ens_z).max(), np.abs(bf_z).max(), 0.5)
    pad = max_abs * 0.6 + 0.5
    for yi, z, p in zip(y + height / 2, ens_z, pair_df["delong_ensemble_p_holm"]):
        ax.text(
            z + (0.08 if z >= 0 else -0.08),
            yi,
            f"p={p:.3f}",
            va="center",
            ha="left" if z >= 0 else "right",
            fontsize=7,
        )
    for yi, z, p in zip(y - height / 2, bf_z, pair_df["delong_bestfold_p_holm"]):
        ax.text(
            z + (0.08 if z >= 0 else -0.08),
            yi,
            f"p={p:.3f}",
            va="center",
            ha="left" if z >= 0 else "right",
            fontsize=7,
        )

    ax.axvline(0, color="#c3c2b7", linewidth=1, linestyle="--")
    ax.set_xlim(-max_abs - pad, max_abs + pad)
    ax.set_yticks(y)
    ax.set_yticklabels(pair_df["pair_id"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("DeLong z-statistic (A - B)")
    ax.set_title("DeLong test, ensemble vs. best-fold predictions (clip-level)", fontsize=10)

    legend_elems = [
        mpatches.Patch(facecolor="white", edgecolor="#0b0b0b", label="Ensemble"),
        mpatches.Patch(facecolor="white", edgecolor="#0b0b0b", hatch="//", label="Best fold"),
        mpatches.Patch(facecolor=STATUS_SIGNIFICANT, label="Holm-adjusted p < 0.05"),
        mpatches.Patch(facecolor=STATUS_NEUTRAL, label="Holm-adjusted p >= 0.05"),
    ]
    ax.legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=7.5,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_grouped_metrics(ax, names, metric_values: dict, ci_values=None, thresholds=None) -> None:
    """Draw one grouped-metrics panel onto `ax`"""
    metric_names = list(metric_values.keys())
    n_models = len(names)
    x = np.arange(len(metric_names))
    width = 0.8 / n_models
    for i, name in enumerate(names):
        values = [metric_values[m][i] for m in metric_names]
        yerr = None
        if ci_values is not None:
            lo_list, hi_list, has_any = [], [], False
            for m in metric_names:
                ci = ci_values.get(m, [None] * n_models)[i]
                if ci is not None:
                    has_any = True
                    lo_list.append(max(metric_values[m][i] - ci[0], 0.0))
                    hi_list.append(max(ci[1] - metric_values[m][i], 0.0))
                else:
                    lo_list.append(0.0)
                    hi_list.append(0.0)
            yerr = [lo_list, hi_list] if has_any else None
        display_name = config_label(name)
        label = (
            display_name if thresholds is None else f"{display_name}  (thr={thresholds[name]:.2f})"
        )
        ax.bar(
            x + i * width,
            values,
            width,
            label=label,
            color=config_color(name),
            yerr=yerr,
            capsize=3,
        )
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([metric_label(m) for m in metric_names], rotation=0)
    ax.set_ylim(0, 1.05)


def _plot_two_level_grouped_metrics(
    names, frame_values: dict, frame_ci: dict, clip_values: dict, clip_ci: dict, title, output_path
) -> None:
    """Frame level on top, clip level on the bottom
    Both panels carry a 95% BCa bootstrap CI"""
    fig, axes = plt.subplots(
        2, 1, figsize=(max(11, len(METRICS) * 2.2), 11.6), sharex=True, sharey=True
    )
    _draw_grouped_metrics(axes[0], names, frame_values, frame_ci)
    axes[0].set_title("Frame level (95% BCa CI)", fontsize=11, fontweight="bold")
    _draw_grouped_metrics(axes[1], names, clip_values, clip_ci)
    axes[1].set_title(
        "Clip level, mean-probability aggregation (95% BCa CI)", fontsize=11, fontweight="bold"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=True
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_metrics(df: pd.DataFrame, title: str, output_path) -> None:
    """Ensemble metrics per configuration (rows of the ensemble table), frame level on
    top and clip level below, with their 95% BCa CIs."""

    def _ci(suffix):
        return {
            m: list(zip(df[f"{m}{suffix}_ci_bca_lower"], df[f"{m}{suffix}_ci_bca_upper"]))
            for m in METRICS
        }

    _plot_two_level_grouped_metrics(
        df["config"].tolist(),
        {m: df[f"{m}_frame"].tolist() for m in METRICS},
        _ci("_frame"),
        {m: df[m].tolist() for m in METRICS},
        _ci(""),
        title,
        output_path,
    )


def plot_pairwise_matrix(names: list[str], bootstrap_matrix_df: pd.DataFrame, output_path) -> None:
    """Full pairwise patient-clustered bootstrap matrix (95% BCa): heatmap
    (upper triangle, green/pale-red = CI excludes/includes 0) + a summary
    table sized to fit every pair. `names` = raw config keys in matrix order.
    """
    p_matrix_boot = pd.DataFrame(np.nan, index=names, columns=names)
    for _, r in bootstrap_matrix_df.iterrows():
        p_matrix_boot.loc[r["config_a"], r["config_b"]] = 0.0 if r["significant"] else 1.0
    names = (
        p_matrix_boot.index.tolist()
    )  # RAW keys -- join key into bootstrap_matrix_df, do not map
    labels = config_labels(names)  # display only, same order as names
    n = len(names)
    n_pairs = len(bootstrap_matrix_df)
    max_name_len = max((len(s) for s in labels), default=8)
    fig_width = max(14, 9 + max_name_len * 0.35)
    fig_height = max(5, n * 0.5, 1.3 + n_pairs * 0.22)

    fig, (ax_heat, ax_table) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    diag_gray = mcolors.to_rgba("#e1e0d9")
    sig_color = mcolors.to_rgba(STATUS_SIGNIFICANT)
    not_sig_color = mcolors.to_rgba(STATUS_NOT_SIGNIFICANT)
    colors = np.ones((n, n, 4))
    for i in range(n):
        for j in range(n):
            if i >= j or pd.isna(p_matrix_boot.iloc[i, j]):
                colors[i, j] = diag_gray
            else:
                colors[i, j] = sig_color if p_matrix_boot.iloc[i, j] < 0.5 else not_sig_color
    ax_heat.imshow(colors, aspect="auto")
    for i in range(n):
        for j in range(i + 1, n):
            if pd.isna(p_matrix_boot.iloc[i, j]):
                continue
            row = bootstrap_matrix_df[
                (bootstrap_matrix_df["config_a"] == names[i])
                & (bootstrap_matrix_df["config_b"] == names[j])
            ].iloc[0]
            ax_heat.text(
                j,
                i,
                f"{row['delta_auroc']:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if row["significant"] else "normal",
                color="#0b0b0b",
            )
        ax_heat.text(i, i, "-", ha="center", va="center", fontsize=10, color=STATUS_NEUTRAL)
    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(labels, fontsize=9)
    ax_heat.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.5)
    ax_heat.tick_params(which="minor", length=0)
    ax_heat.set_title(
        "Heldout pairwise ΔAUROC (patient-clustered bootstrap, 95% BCa CI)",
        fontsize=11,
    )
    legend_elems = [
        mpatches.Patch(facecolor=STATUS_SIGNIFICANT, label="CI excludes 0 (significant)"),
        mpatches.Patch(facecolor=STATUS_NOT_SIGNIFICANT, label="CI includes 0 (not significant)"),
    ]
    ax_heat.legend(
        handles=legend_elems,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    ax_table.axis("off")
    y = 0.97
    ax_table.text(
        0.05,
        y,
        "Patient-clustered bootstrap, all pairs",
        fontsize=11,
        fontweight="bold",
        transform=ax_table.transAxes,
        va="top",
    )
    y -= 0.06
    headers = ["Model A", "Model B", "ΔAUROC", "95% BCa CI", "sig."]
    name_col_width = min(0.30, 0.09 + max_name_len * 0.011)
    col_x = [
        0.00,
        name_col_width,
        2 * name_col_width,
        2 * name_col_width + 0.14,
        2 * name_col_width + 0.34,
    ]
    for hdr, x in zip(headers, col_x):
        ax_table.text(
            x + 0.02,
            y,
            hdr,
            color=STATUS_NEUTRAL,
            fontsize=8,
            fontweight="bold",
            transform=ax_table.transAxes,
            va="top",
        )
    y -= 0.04
    ax_table.axhline(y, color="#e1e0d9", linewidth=0.8, xmin=0.02, xmax=0.98)
    y -= 0.01

    row_step = y / max(n_pairs, 1)
    for _, row in bootstrap_matrix_df.iterrows():
        color = STATUS_SIGNIFICANT if row["significant"] else "#0b0b0b"
        vals = [
            config_label(row["config_a"]),
            config_label(row["config_b"]),
            f"{row['delta_auroc']:+.4f}",
            f"({row['ci_lower']:+.4f}, {row['ci_upper']:+.4f})",
            "Yes" if row["significant"] else "No",
        ]
        for val, x in zip(vals, col_x):
            ax_table.text(
                x + 0.02, y, val, color=color, fontsize=7.5, transform=ax_table.transAxes, va="top"
            )
        y -= row_step

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fold_dispersion_table(df: pd.DataFrame, output_path) -> None:
    """Render the frame/clip fold-dispersion table (mean +/- std across the 5
    individually-trained fold models, not the ensemble)"""

    def _level_table(level: str) -> tuple[list, list]:
        header = ["Config"] + [metric_label(m) for m in METRICS]
        cell_text = []
        for _, row in df.iterrows():
            cells = [config_label(row["config"], short=False)]
            for m in METRICS:
                cells.append(f"{row[f'{m}_{level}_mean']:.3f} +/- {row[f'{m}_{level}_std']:.3f}")
            cell_text.append(cells)
        return header, cell_text

    header, cells_frame = _level_table("frame")
    _, cells_clip = _level_table("clip")

    lengths = text_lengths([header] + cells_frame + cells_clip)
    fig_w = max(11.0, 1.0 + 0.145 * sum(lengths))
    fig_h = max(3.4, 0.9 + 0.45 * len(df) * 2)

    fig, (ax_frame, ax_clip) = plt.subplots(2, 1, figsize=(fig_w, fig_h))
    for ax, cell_text, level_title in (
        (ax_frame, cells_frame, "Frame level"),
        (ax_clip, cells_clip, "Clip level"),
    ):
        draw_config_table(
            ax, header, cell_text, df["config"], column_widths(lengths), fontsize=7.5, row_scale=1.6
        )
        ax.set_title(level_title, fontsize=10.5, fontweight="bold", pad=8)

    fig.suptitle(
        "Heldout test-set performance, mean +/- std across the 5 CV fold models",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_roc(
    configs: dict[str, ConfigPredictions], cohort: HeldoutCohort, output_path
) -> None:
    """Ensemble ROC curves, frame level (left) and clip level (right)."""
    clip_labels, manifest_frame_labels = cohort.clip_labels, cohort.frame_labels
    roc_data_clip = []
    for model_key, data in configs.items():
        probs = data.ensemble_clip_probs
        fpr, tpr, _ = roc_curve(clip_labels, probs)
        roc_data_clip.append(
            {
                "name": config_label(model_key),
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc_score(clip_labels, probs),
                "color": config_color(model_key),
            }
        )
    roc_data_frame = []
    for model_key, data in configs.items():
        probs = data.ensemble_frame_probs
        fpr, tpr, _ = roc_curve(manifest_frame_labels, probs)
        roc_data_frame.append(
            {
                "name": config_label(model_key),
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc_score(manifest_frame_labels, probs),
                "color": config_color(model_key),
            }
        )
    fig, (ax_frame, ax_clip) = plt.subplots(1, 2, figsize=(15.5, 8.2))
    plot_roc_curves(roc_data_frame, title="Frame level", ax=ax_frame)
    plot_roc_curves(roc_data_clip, title="Clip level (mean-probability aggregation)", ax=ax_clip)
    fig.suptitle("Heldout ensemble ROC curves", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
