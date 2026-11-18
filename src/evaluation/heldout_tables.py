"""
src/evaluation/heldout_tables.py
--------------------------------
Every table of the held-out analysis, from the configurations' predictions.

Analysis tables (CSV values rounded to 4 decimals)
    per_fold_table          each fold model's frame/clip metrics
    fold_dispersion_table   mean / std across the 5 fold models
    ensemble_table          5-fold ensemble, point + 95% BCa CI (patient-clustered)
    pair_table              pre-specified method / corpus effects (PAIRS)
    pairwise_matrix         all configuration pairs (exploratory, uncorrected)
    patient_error_table     per-patient clip mean absolute error (Friedman blocks)

Manuscript tables (built from unrounded values)
    paper_table1            ensemble metrics with CI, frame and clip level
    paper_table_s1          fold mean +/- std, frame and clip level
    paper_comparisons       every DeltaAUROC quoted in the text, oriented A - B
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config.models import get_model_entry
from src.evaluation.bootstrap import (
    delta_auroc_statistic,
    flip_delta,
    metric_statistic,
    patient_clustered_bootstrap,
)
from src.evaluation.delong import delong_test
from src.evaluation.heldout import METRICS, ConfigPredictions, HeldoutCohort
from src.evaluation.rank_tests import holm_bonferroni

Configs = dict[str, ConfigPredictions]

# Pre-specified architecture-matched comparisons, DeltaAUROC = A - B.
PAIRS: list[dict] = [
    {
        "pair_id": "method_resnet50",
        "effect": "method",
        "architecture": "ResNet-50",
        "config_a": "resnet50_imagenet_sup",
        "config_b": "resnet50_imagenet",
    },
    {
        "pair_id": "method_vitb16",
        "effect": "method",
        "architecture": "ViT-Base/16",
        "config_a": "vitb16_imagenet_sup",
        "config_b": "vitb16_imagenet",
    },
    {
        "pair_id": "method_vits16",
        "effect": "method",
        "architecture": "ViT-Small/16",
        "config_a": "vits16_imagenet_hf",
        "config_b": "vits16_imagenet",
    },
    {
        "pair_id": "corpus_resnet50",
        "effect": "corpus",
        "architecture": "ResNet-50",
        "config_a": "resnet50_imagenet",
        "config_b": "resnet50_gastronet",
    },
    {
        "pair_id": "corpus_vits16",
        "effect": "corpus",
        "architecture": "ViT-Small/16",
        "config_a": "vits16_imagenet",
        "config_b": "vits16_gastronet",
    },
]


def _config_info(model_key: str) -> dict:
    entry = get_model_entry(model_key)
    return {
        "config": model_key,
        "architecture": entry.architecture,
        "pretrain_data": entry.pretrain_data,
        "pretrain_method": entry.pretrain_method,
    }


def _boot(labels, patient_ids, statistic, n_bootstrap: int, seed: int, alpha: float) -> dict:
    return patient_clustered_bootstrap(
        labels, patient_ids, statistic, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha
    )


# ---------------------------------------------------------------------------
# Fold-level tables
# ---------------------------------------------------------------------------


def per_fold_table(configs: Configs) -> pd.DataFrame:
    rows = []
    for model_key, data in configs.items():
        info = _config_info(model_key)
        folds = zip(data.fold_metrics, data.fold_thresholds, data.fold_metrics_frame)
        for i, (m, thr, m_frame) in enumerate(folds, start=1):
            rows.append(
                {
                    **info,
                    "fold": i,
                    "threshold": round(thr, 4),
                    **{k: round(v, 4) for k, v in m.items()},
                    **{f"{k}_frame": round(v, 4) for k, v in m_frame.items()},
                }
            )
    return pd.DataFrame(rows)


def _fold_dispersion(data: ConfigPredictions) -> dict:
    """Unrounded {f"{metric}_{level}_{mean|std}": value} across the fold models."""
    out = {}
    for level, fold_metrics in (("frame", data.fold_metrics_frame), ("clip", data.fold_metrics)):
        for metric in METRICS:
            vals = np.array([fm[metric] for fm in fold_metrics], dtype=float)
            vals = vals[~np.isnan(vals)]
            out[f"{metric}_{level}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            out[f"{metric}_{level}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def fold_dispersion_table(configs: Configs) -> pd.DataFrame:
    """Mean / std across the 5 individually trained fold models (not the ensemble)."""
    rows = []
    for model_key, data in configs.items():
        row = {**_config_info(model_key), "n_folds": len(data.fold_metrics)}
        row.update({k: round(v, 4) for k, v in _fold_dispersion(data).items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ensemble and comparisons
# ---------------------------------------------------------------------------


class EnsembleResults(NamedTuple):
    table: pd.DataFrame
    clip: dict  # model_key -> metric -> bootstrap result
    frame: dict  # model_key -> metric -> bootstrap result


def ensemble_table(
    configs: Configs, cohort: HeldoutCohort, n_bootstrap: int, seed: int, alpha: float
) -> EnsembleResults:
    """Ensemble metrics at the mean CV threshold, with patient-clustered BCa CIs."""
    clip_boot, frame_boot, rows = {}, {}, []
    for model_key, data in configs.items():
        thr, frame_thr = data.cv_mean_clip_threshold, data.cv_mean_threshold
        clip_boot[model_key] = {
            m: _boot(
                cohort.clip_labels,
                cohort.clip_patient_ids,
                metric_statistic(m, cohort.clip_labels, data.ensemble_clip_probs, thr),
                n_bootstrap,
                seed,
                alpha,
            )
            for m in METRICS
        }
        frame_boot[model_key] = {
            m: _boot(
                cohort.frame_labels,
                cohort.frame_patient_ids,
                metric_statistic(m, cohort.frame_labels, data.ensemble_frame_probs, frame_thr),
                n_bootstrap,
                seed,
                alpha,
            )
            for m in METRICS
        }
        best_fold_auroc = (
            float(roc_auc_score(cohort.clip_labels, data.best_fold_clip_probs))
            if len(np.unique(cohort.clip_labels)) > 1
            else float("nan")
        )
        row = {
            **_config_info(model_key),
            "threshold": round(thr, 4),
            "threshold_frame": round(frame_thr, 4),
            "best_fold_auroc": round(best_fold_auroc, 4),
        }
        for m in METRICS:
            for suffix, r in (("", clip_boot[model_key][m]), ("_frame", frame_boot[model_key][m])):
                row[f"{m}{suffix}"] = round(r["point"], 4)
                row[f"{m}{suffix}_ci_bca_lower"] = round(r["ci_bca"][0], 4)
                row[f"{m}{suffix}_ci_bca_upper"] = round(r["ci_bca"][1], 4)
                row[f"{m}{suffix}_n_bootstrap_dropped"] = r["n_dropped"]
        rows.append(row)

        a, af = clip_boot[model_key]["auroc"], frame_boot[model_key]["auroc"]
        print(
            f"  {model_key:28s} thr={thr:.3f}  AUROC={a['point']:.4f}  "
            f"BCa=({a['ci_bca'][0]:.4f}-{a['ci_bca'][1]:.4f})  "
            f"dropped={a['n_dropped']}/{n_bootstrap}  |  "
            f"frame thr={frame_thr:.3f} AUROC={af['point']:.4f}  "
            f"BCa=({af['ci_bca'][0]:.4f}-{af['ci_bca'][1]:.4f})"
        )
    return EnsembleResults(pd.DataFrame(rows), clip_boot, frame_boot)


def _delta_boot(configs, cohort, a, b, n_bootstrap, seed, alpha) -> dict:
    statistic = delta_auroc_statistic(
        cohort.clip_labels, configs[a].ensemble_clip_probs, configs[b].ensemble_clip_probs
    )
    return _boot(cohort.clip_labels, cohort.clip_patient_ids, statistic, n_bootstrap, seed, alpha)


def pair_table(
    configs: Configs, cohort: HeldoutCohort, n_bootstrap: int, seed: int, alpha: float
) -> tuple[pd.DataFrame, dict]:
    """PAIRS: ensemble DeltaAUROC with BCa CI (primary), plus ensemble and best-fold
    DeLong with Holm correction (clip level, not clustered, kept for traceability)."""
    rows, boots, ens_p, best_p = [], {}, [], []
    labels = cohort.clip_labels
    for pair in PAIRS:
        a, b = pair["config_a"], pair["config_b"]
        if a not in configs or b not in configs:
            print(f"  [warn] {pair['pair_id']}: missing config(s), skipping.")
            continue
        boot = _delta_boot(configs, cohort, a, b, n_bootstrap, seed, alpha)
        boots[pair["pair_id"]] = boot
        auc_a, auc_b, z_ens, p_ens = delong_test(
            labels, configs[a].ensemble_clip_probs, configs[b].ensemble_clip_probs
        )
        auc_a_bf, auc_b_bf, z_bf, p_bf = delong_test(
            labels, configs[a].best_fold_clip_probs, configs[b].best_fold_clip_probs
        )
        ens_p.append(p_ens)
        best_p.append(p_bf)
        rows.append(
            {
                **pair,
                "auroc_a_ensemble": round(auc_a, 4),
                "auroc_b_ensemble": round(auc_b, 4),
                "delta_auroc_ensemble": round(boot["point"], 4),
                "boot_ci_bca_lower": round(boot["ci_bca"][0], 4),
                "boot_ci_bca_upper": round(boot["ci_bca"][1], 4),
                "n_bootstrap_dropped": boot["n_dropped"],
                "delong_ensemble_z": round(z_ens, 3),
                "delong_ensemble_p": round(p_ens, 4),
                "auroc_a_bestfold": round(auc_a_bf, 4),
                "auroc_b_bestfold": round(auc_b_bf, 4),
                "delta_auroc_bestfold": round(auc_a_bf - auc_b_bf, 4),
                "delong_bestfold_z": round(z_bf, 3),
                "delong_bestfold_p": round(p_bf, 4),
            }
        )
        print(
            f"  {pair['pair_id']:20s} DeltaAUROC={boot['point']:+.4f}  "
            f"BCa=({boot['ci_bca'][0]:+.4f},{boot['ci_bca'][1]:+.4f})  "
            f"DeLong(ens) p={p_ens:.4f}  DeLong(best) p={p_bf:.4f}"
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["delong_ensemble_p_holm"] = np.round(holm_bonferroni(ens_p), 4)
        df["delong_bestfold_p_holm"] = np.round(holm_bonferroni(best_p), 4)
    return df, boots


def pairwise_matrix(
    configs: Configs, cohort: HeldoutCohort, n_bootstrap: int, seed: int, alpha: float
) -> tuple[pd.DataFrame, dict]:
    """Every configuration pair (exploratory: not pre-specified, no multiplicity
    correction). Returns the table sorted by CI lower bound and the raw results
    keyed by (config_a, config_b)."""
    rows, boots = [], {}
    for a, b in itertools.combinations(list(configs), 2):
        boot = _delta_boot(configs, cohort, a, b, n_bootstrap, seed, alpha)
        boots[(a, b)] = boot
        lo, hi = boot["ci_bca"]
        rows.append(
            {
                "config_a": a,
                "config_b": b,
                "delta_auroc": round(boot["point"], 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "significant": bool(lo > 0 or hi < 0),
            }
        )
    df = pd.DataFrame(rows).sort_values("ci_lower").reset_index(drop=True)
    return df, boots


def patient_error_table(configs: Configs, cohort: HeldoutCohort) -> pd.DataFrame:
    """Rows = patients (Friedman blocks), columns = configurations, values = mean
    absolute error of the ensemble clip probabilities. Defined for every patient,
    unlike a per-patient AUROC (2/19 patients have a single clip class)."""
    rows = []
    for patient in np.unique(cohort.clip_patient_ids):
        mask = cohort.clip_patient_ids == patient
        row = {"patient_id": patient}
        for model_key, data in configs.items():
            row[model_key] = float(
                np.mean(np.abs(cohort.clip_labels[mask] - data.ensemble_clip_probs[mask]))
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("patient_id")


# ---------------------------------------------------------------------------
# Manuscript tables
# ---------------------------------------------------------------------------

PAPER_ROW_ORDER = [
    "efficientnetb0",
    "resnet50_imagenet_sup",
    "resnet50_imagenet",
    "resnet50_gastronet",
    "vitb16_imagenet_sup",
    "vitb16_imagenet",
    "vits16_imagenet_hf",
    "vits16_imagenet",
    "vits16_gastronet",
]
PAPER_METRICS = {"auroc": "AUROC", "sensitivity": "Sens", "specificity": "Spec", "f1": "F1"}


class Comparison(NamedTuple):
    label: str
    a: str
    b: str
    prespecified: bool


# Every DeltaAUROC quoted in the text, oriented as written: A - B.
PAPER_COMPARISONS = [
    Comparison("Corpus, ResNet-50", "resnet50_gastronet", "resnet50_imagenet", True),
    Comparison("Corpus, ViT-S/16", "vits16_gastronet", "vits16_imagenet", True),
    Comparison("Method, ResNet-50", "resnet50_imagenet", "resnet50_imagenet_sup", True),
    Comparison("Method, ViT-B/16", "vitb16_imagenet", "vitb16_imagenet_sup", True),
    Comparison("Method, ViT-S/16", "vits16_imagenet", "vits16_imagenet_hf", True),
    Comparison("GastroNet RN50 vs ViT-S", "resnet50_gastronet", "vits16_gastronet", False),
    Comparison("GastroNet ViT-S vs EffNet", "vits16_gastronet", "efficientnetb0", False),
    Comparison("GastroNet RN50 vs EffNet", "resnet50_gastronet", "efficientnetb0", False),
    Comparison("ImageNet-1K ViT-S vs EffNet", "vits16_imagenet_hf", "efficientnetb0", False),
]


def _paper_row_header(model_key: str) -> dict:
    """Backbone + pretraining as printed in the manuscript (dagger = self-supervised)."""
    entry = get_model_entry(model_key)
    pretraining = entry.pretrain_data
    if entry.pretrain_method.startswith("Self-sup"):
        method = entry.pretrain_method.split("(")[-1].rstrip(")")
        pretraining = f"{entry.pretrain_data} ({method}) †"
    return {"Backbone": entry.architecture, "Pretraining": pretraining}


def _paper_rows(model_keys, cell) -> pd.DataFrame:
    rows = []
    for key in (k for k in PAPER_ROW_ORDER if k in model_keys):
        row = _paper_row_header(key)
        for level in ("frame", "clip"):
            for metric, name in PAPER_METRICS.items():
                row[f"{level.capitalize()} {name}"] = cell(key, level, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def paper_table1(ensemble: EnsembleResults) -> pd.DataFrame:
    """Ensemble metric (95% BCa CI), the estimand of every DeltaAUROC in the text."""

    def cell(key, level, metric):
        r = (ensemble.clip if level == "clip" else ensemble.frame)[key][metric]
        return f"{r['point']:.3f} ({r['ci_bca'][0]:.3f}–{r['ci_bca'][1]:.3f})"

    return _paper_rows(ensemble.clip, cell)


def paper_table_s1(configs: Configs) -> pd.DataFrame:
    """Mean +/- std across the 5 fold models: training variability on the same
    patients, not sampling uncertainty."""
    disp = {key: _fold_dispersion(data) for key, data in configs.items()}

    def cell(key, level, metric):
        d = disp[key]
        return f"{d[f'{metric}_{level}_mean']:.3f} ±{d[f'{metric}_{level}_std']:.3f}"

    return _paper_rows(configs, cell)


def paper_comparisons(matrix_boots: dict, configs: Configs, cohort: HeldoutCohort) -> pd.DataFrame:
    rows = []
    for c in PAPER_COMPARISONS:
        if (c.a, c.b) in matrix_boots:
            r = matrix_boots[(c.a, c.b)]
            delta, (lo, hi) = r["point"], r["ci_bca"]
        else:
            r = matrix_boots[(c.b, c.a)]
            delta, (lo, hi) = flip_delta(r["point"], r["ci_bca"])
        auc = {
            k: roc_auc_score(cohort.clip_labels, configs[k].ensemble_clip_probs) for k in (c.a, c.b)
        }
        rows.append(
            {
                "comparison": c.label,
                "A": c.a,
                "B": c.b,
                "prespecified": c.prespecified,
                "auroc_A": f"{auc[c.a]:.4f}",
                "auroc_B": f"{auc[c.b]:.4f}",
                "delta_auroc": f"{delta:+.4f}",
                "ci_95_bca": f"{lo:+.4f} to {hi:+.4f}",
                "ci_excludes_0": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(rows)
