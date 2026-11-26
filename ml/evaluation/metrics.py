"""Evaluation metrics used across all FinMind model training.

All metrics are computed from real predictions — nothing is estimated,
clipped to look better, or fabricated. MAPE is reported only where the
denominator is valid (small denominators make MAPE explode; those points
are counted and reported instead of silently dropped).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Regression / forecasting metrics
# ---------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> Tuple[float, int]:
    """Mean absolute percentage error, %.

    Returns (mape_percent, n_excluded) where n_excluded counts points whose
    |true| value is below eps (excluded because the percentage is undefined
    or numerically meaningless there — never silently hidden).
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true)
    valid = denom >= eps
    n_excluded = int((~valid).sum())
    if valid.sum() == 0:
        return float("nan"), n_excluded
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / denom[valid])) * 100.0), n_excluded


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """Full metric bundle for a forecasting/regression model."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mape_pct, n_excluded = mape(y_true, y_pred)
    return {
        "mae": round(mae(y_true, y_pred), 6),
        "rmse": round(rmse(y_true, y_pred), 6),
        "mape_pct": round(mape_pct, 4) if not np.isnan(mape_pct) else None,
        "mape_excluded_points": n_excluded,
        "r2": round(r2_score(y_true, y_pred), 6),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Classification / fraud metrics
# ---------------------------------------------------------------------------
def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Precision / recall / F1 / PR-AUC / ROC-AUC.

    y_score: continuous score (probability or anomaly score where higher =
    more anomalous). If y_pred is None it is derived via `threshold` on
    y_score (for probabilities). For anomaly scores the caller must pass
    y_pred explicitly (IsolationForest uses lower score = more anomalous).
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred, int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics: Dict[str, object] = {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    # AUC metrics need both classes present in y_true.
    if len(np.unique(y_true)) == 2:
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 6)
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 6)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None
    return metrics


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Scan candidate thresholds and return (threshold, best_f1).

    Threshold selection must happen on TRAINING/VALIDATION data only —
    callers must never pass the test set here (see leakage audit).
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    candidates = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)
