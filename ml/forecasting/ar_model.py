"""Autoregressive AR(p) baseline forecaster built on statsmodels AutoReg.

Used as a transparent statistical baseline against the deep-learning
recurrent models. The forecast service selects the best model per business
using walk-forward holdout error – no metric is fabricated.
"""

from __future__ import annotations

from typing import List

import numpy as np

try:
    from statsmodels.tsa.ar_model import AutoReg
except ImportError as exc:  # pragma: no cover
    raise ImportError("statsmodels is required for the AR(p) baseline") from exc


class ARForecaster:
    """AR(p) one-step/recursive forecaster with a deterministic fit."""

    def __init__(self, lags: int = 14):
        if lags < 1:
            raise ValueError("lags must be >= 1")
        self.lags = lags
        self._model = None
        self._res = None

    def fit(self, series: np.ndarray) -> "ARForecaster":
        series = np.asarray(series, dtype=np.float64).ravel()
        if len(series) <= self.lags + 2:
            raise ValueError("not enough observations for AR fit")
        self._model = AutoReg(series, lags=self.lags, old_names=False)
        self._res = self._model.fit()
        return self

    @property
    def params(self) -> np.ndarray:
        return np.asarray(self._res.params, dtype=np.float64)

    def predict_next(self, context: np.ndarray) -> float:
        if self._res is None:
            raise RuntimeError("ARForecaster must be fit before predicting")
        window = np.asarray(context, dtype=np.float64).ravel()[-self.lags:]
        if len(window) < self.lags:
            raise ValueError("context shorter than lag order")
        # y_t = const + sum(phi_i * y_{t-i})
        const = self.params[0]
        phis = self.params[1:]
        return float(const + float(np.dot(phis, window)))

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        window = list(np.asarray(context, dtype=np.float64).ravel())
        out: List[float] = []
        for _ in range(horizon):
            nxt = self.predict_next(np.array(window[-self.lags:]))
            out.append(nxt)
            window.append(nxt)
        return np.array(out)


class PooledARForecaster:
    """AR(p) fitted by OLS over (window -> next value) pairs pooled from
    MULTIPLE series (each pre-scaled with its own training statistics).

    This is the globally-saved AR(14) artifact: one coefficient set usable
    at inference on any series without retraining. Windows never cross a
    series boundary (handled by the caller passing per-series windows).
    """

    def __init__(self, lags: int = 14):
        if lags < 1:
            raise ValueError("lags must be >= 1")
        self.lags = lags
        self.intercept_ = 0.0
        self.coef_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PooledARForecaster":
        """X: (N, lags) context windows, y: (N,) next values (scaled)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or X.shape[1] != self.lags:
            raise ValueError(f"X must have shape (N, {self.lags})")
        design = np.hstack([np.ones((len(X), 1)), X])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = np.asarray(beta[1:], dtype=np.float64)
        return self

    def predict_next(self, context: np.ndarray) -> float:
        window = np.asarray(context, dtype=np.float64).ravel()[-self.lags:]
        if len(window) < self.lags:
            raise ValueError("context shorter than lag order")
        return float(self.intercept_ + float(np.dot(self.coef_, window)))

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        window = list(np.asarray(context, dtype=np.float64).ravel())
        out: List[float] = []
        for _ in range(horizon):
            nxt = self.predict_next(np.array(window[-self.lags:]))
            out.append(nxt)
            window.append(nxt)
        return np.array(out)
