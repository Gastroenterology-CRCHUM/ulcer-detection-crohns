"""
src/noninformative/model.py
===========================
Random Forest classifier for non-informative frame detection.

Follows the paper:
  - Feature fusion: 37 hand-crafted + 2048 Inception-v3 bottleneck = 2085 features
  - Classifier: Random Forest (automatic feature selection)
  - Evaluation: frame-level + sample-level aggregation

Pipeline
--------
    train()   → fits scaler + RF, saves artefacts
    evaluate() → full metrics with bootstrap CI
    load()    → restore trained pipeline from disk
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_RF_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
    "class_weight": "balanced",
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class NonInformativeClassifier:
    """
    Scaler + Random Forest pipeline for non-informative frame detection.

    Attributes
    ----------
    scaler    : StandardScaler fitted on training features.
    rf        : Trained RandomForestClassifier.
    threshold : Decision threshold (tuned on val set if called).
    feature_importances : pd.Series with named importances (after training).
    """

    def __init__(self, rf_params: dict | None = None):
        self.rf_params = {**DEFAULT_RF_PARAMS, **(rf_params or {})}
        self.scaler: StandardScaler | None = None
        self.rf: RandomForestClassifier | None = None
        self.threshold: float = 0.5
        self.feature_importances: pd.Series | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
        verbose: bool = True,
    ) -> NonInformativeClassifier:
        """
        Fit scaler + Random Forest on training data.

        Args:
            X_train       : (N, D) feature matrix.
            y_train       : (N,) binary labels — 1=Informative, 0=Non-Informative.
            feature_names : Optional list of D feature names (for importances).
            verbose       : Print training summary.

        Returns:
            self
        """
        self._feature_names = feature_names

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.rf = RandomForestClassifier(**self.rf_params)
        self.rf.fit(X_scaled, y_train)

        if feature_names and len(feature_names) == X_train.shape[1]:
            self.feature_importances = pd.Series(
                self.rf.feature_importances_, index=feature_names
            ).sort_values(ascending=False)

        if verbose:
            n1 = (y_train == 1).sum()
            n0 = (y_train == 0).sum()
            print(
                f"[RF] Trained on {len(y_train)} samples  "
                f"(Informative: {n1}, Non-Informative: {n0})"
            )
            print(
                f"[RF] n_estimators={self.rf_params['n_estimators']}, "
                f"max_features={self.rf_params['max_features']}"
            )
            if self.feature_importances is not None:
                print("[RF] Top-10 features:")
                for name, imp in self.feature_importances.head(10).items():
                    print(f"     {name:<40s}  {imp:.4f}")

        return self

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def tune_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1",
        n_steps: int = 50,
        verbose: bool = True,
    ) -> float:
        """
        Sweep thresholds on the validation set and pick the best one.

        Args:
            X_val   : Validation features.
            y_val   : Validation labels.
            metric  : 'f1' | 'balanced_accuracy'.
            n_steps : Number of threshold candidates.

        Returns:
            Best threshold (also stored as self.threshold).
        """
        probs = self.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.05, 0.95, n_steps)
        best_score, best_t = -1.0, 0.5

        for t in thresholds:
            preds = (probs >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_val, preds, zero_division=0)
            elif metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_val, preds)
            else:
                score = accuracy_score(y_val, preds)

            if score > best_score:
                best_score, best_t = score, float(t)

        self.threshold = best_t
        if verbose:
            print(f"[RF] Optimal threshold ({metric}): {best_t:.3f}  →  score={best_score:.4f}")
        return best_t

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (N, 2)."""
        self._check_fitted()
        return self.rf.predict_proba(self.scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions using self.threshold."""
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_single(self, x: np.ndarray) -> tuple[int, float]:
        """
        Predict a single feature vector.

        Returns:
            (label, probability_of_noninformative)
        """
        p = self.predict_proba(x.reshape(1, -1))[0, 1]
        return int(p >= self.threshold), float(p)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_bootstrap: int = 2000,
    ) -> dict:
        """
        Compute full evaluation metrics with bootstrap 95% CIs.

        Returns:
            Dict with frame-level metrics and per-class report.
        """
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        def _bci(fn, yt, yp, n=n_bootstrap):
            scores = []
            size = len(yt)
            for _ in range(n):
                idx = np.random.choice(size, size, replace=True)
                if len(np.unique(yt[idx])) < 2:
                    continue
                try:
                    scores.append(fn(yt[idx], yp[idx]))
                except Exception:
                    continue
            if not scores:
                return float("nan"), float("nan")
            return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

        f1_lo, f1_hi = _bci(lambda a, b: f1_score(a, b, zero_division=0), y_test, preds)
        auc_lo, auc_hi = _bci(roc_auc_score, y_test, probs)
        acc_lo, acc_hi = _bci(accuracy_score, y_test, preds)

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn_v, tp = cm.ravel()
        sensitivity = tp / (tp + fn_v) if (tp + fn_v) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        results = {
            "threshold": self.threshold,
            "f1": f1_score(y_test, preds, zero_division=0),
            "f1_ci": (f1_lo, f1_hi),
            "roc_auc": roc_auc_score(y_test, probs),
            "roc_auc_ci": (auc_lo, auc_hi),
            "accuracy": accuracy_score(y_test, preds),
            "accuracy_ci": (acc_lo, acc_hi),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": cm,
            "report": classification_report(
                y_test,
                preds,
                target_names=["Non-Informative", "Informative"],
                output_dict=True,
            ),
            "predictions": preds,
            "probabilities": probs,
        }

        print("=" * 55)
        print("EVALUATION RESULTS")
        print("=" * 55)
        print(f"  F1          : {results['f1']:.4f}  (95% CI {f1_lo:.4f}–{f1_hi:.4f})")
        print(f"  AUROC       : {results['roc_auc']:.4f}  (95% CI {auc_lo:.4f}–{auc_hi:.4f})")
        print(f"  Accuracy    : {results['accuracy']:.4f}  (95% CI {acc_lo:.4f}–{acc_hi:.4f})")
        print(f"  Sensitivity : {sensitivity:.4f}")
        print(f"  Specificity : {specificity:.4f}")
        print("=" * 55)
        print(
            classification_report(
                y_test,
                preds,
                target_names=["Non-Informative", "Informative"],
            )
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "rf": self.rf,
                    "threshold": self.threshold,
                    "rf_params": self.rf_params,
                    "feature_names": self._feature_names,
                },
                f,
            )
        print(f"[RF] Pipeline saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> NonInformativeClassifier:
        """Restore a fitted pipeline from disk."""
        with open(Path(path), "rb") as f:
            state = pickle.load(f)
        obj = cls(rf_params=state["rf_params"])
        obj.scaler = state["scaler"]
        obj.rf = state["rf"]
        obj.threshold = state["threshold"]
        obj._feature_names = state.get("feature_names")
        if obj._feature_names and obj.rf is not None:
            obj.feature_importances = pd.Series(
                obj.rf.feature_importances_, index=obj._feature_names
            ).sort_values(ascending=False)
        print(f"[RF] Pipeline loaded ← {path}")
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.rf is None or self.scaler is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
