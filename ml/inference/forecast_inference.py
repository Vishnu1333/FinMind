"""Artifact-backed forecasting inference (Phase 10).

This module is the ONLY path from a saved model artifact to a prediction:

    business daily series (validation)
        -> z-score preprocessing using the business's own observed history
        -> loaded artifact (weights from data/models/finmind_*/)
        -> recursive multi-step forecast
        -> inverse transform
        -> structured response

The selected artifact is read from data/models/registry.json (entries are
produced by the training scripts, never hand-written). Its checksum is
verified against the registry before first use. NOTHING is trained here —
a user request only ever performs forward passes.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Forward pass for a saved recurrent artifact (rnn / lstm / gru)
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class _RecurrentArtifact:
    """Reconstructs the forward pass from raw parameter arrays in the npz."""

    def __init__(self, cell: str, H: int, seq_len: int, Wx, Uh, b, Wy, by):
        self.cell, self.H, self.seq_len = cell, int(H), int(seq_len)
        self.Wx, self.Uh, self.b, self.Wy, self.by = Wx, Uh, b, Wy, by

    def predict_next(self, context: np.ndarray) -> float:
        v = np.asarray(context, dtype=np.float64).ravel()[-self.seq_len:].reshape(1, -1, 1)
        N, T = v.shape[0], v.shape[1]
        h = np.zeros((N, self.H))
        c = np.zeros((N, self.H)) if self.cell == "lstm" else None
        for t in range(T):
            x_t = v[:, t, :].reshape(N, 1)
            pre = x_t @ self.Wx + h @ self.Uh + self.b
            H = self.H
            if self.cell == "lstm":
                i = _sigmoid(pre[:, 0:H])
                f = _sigmoid(pre[:, H:2 * H])
                o = _sigmoid(pre[:, 2 * H:3 * H])
                g = np.tanh(pre[:, 3 * H:4 * H])
                c = f * c + i * g
                h = o * np.tanh(c)
            elif self.cell == "gru":
                z = _sigmoid(pre[:, 0:H])
                r = _sigmoid(pre[:, H:2 * H])
                pre_n = x_t @ self.Wx[:, 2 * H:3 * H] + (r * h) @ self.Uh[:, 2 * H:3 * H] + self.b[:, 2 * H:3 * H]
                n = np.tanh(pre_n)
                h = (1 - z) * n + z * h
            else:  # rnn
                h = np.tanh(pre[:, 0:H])
        return float((h @ self.Wy + self.by)[0, 0])

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        window = list(np.asarray(context, dtype=np.float64).ravel())
        out = []
        for _ in range(horizon):
            nxt = self.predict_next(np.array(window[-self.seq_len:]))
            out.append(nxt)
            window.append(nxt)
        return np.asarray(out)


class _PooledARArtifact:
    def __init__(self, intercept: float, coef: np.ndarray, lags: int):
        self.intercept, self.coef, self.lags = float(intercept), np.asarray(coef), int(lags)

    def predict_next(self, context: np.ndarray) -> float:
        window = np.asarray(context, dtype=np.float64).ravel()[-self.lags:]
        return float(self.intercept + float(np.dot(self.coef, window)))

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        window = list(np.asarray(context, dtype=np.float64).ravel())
        out = []
        for _ in range(horizon):
            nxt = self.predict_next(np.array(window[-self.lags:]))
            out.append(nxt)
            window.append(nxt)
        return np.asarray(out)


class _NaiveArtifact:
    def __init__(self, window: int = 1):
        self.window = window

    def predict_next(self, context: np.ndarray) -> float:
        return float(np.asarray(context)[-1])

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, float(np.asarray(context)[-1]))


class _MovingAverageArtifact:
    def __init__(self, window: int = 7):
        self.window = window

    def predict_next(self, context: np.ndarray) -> float:
        return float(np.asarray(context)[-self.window:].mean())

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        window = list(np.asarray(context, dtype=np.float64).ravel())
        out = []
        for _ in range(horizon):
            nxt = float(np.mean(window[-self.window:]))
            out.append(nxt)
            window.append(nxt)
        return np.asarray(out)


class ArtifactNotAvailableError(RuntimeError):
    """Raised when no validated forecasting artifact exists on disk."""


# ---------------------------------------------------------------------------
# Registry-driven loader (process-level singleton, thread-safe)
# ---------------------------------------------------------------------------
class ForecastModelManager:
    """Loads the registry-selected forecasting artifact exactly once."""

    def __init__(self, registry_path: str = "data/models/registry.json"):
        self.registry_path = registry_path
        self._lock = threading.Lock()
        self._loaded: Optional[Dict] = None

    def _read_registry(self) -> Dict:
        if not os.path.exists(self.registry_path):
            raise ArtifactNotAvailableError(
                "Model registry not found at data/models/registry.json — "
                "run the training pipeline first (ml.training.train_forecast_models)."
            )
        with open(self.registry_path) as f:
            return json.load(f)

    def _checksum(self, path: str) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def load(self) -> Dict:
        """Return {'name','model','seq_len','registry_entry','candidates'}
        for the selected forecasting artifact."""
        with self._lock:
            if self._loaded is not None:
                return self._loaded

            registry = self._read_registry()
            forecasting = [
                e for e in registry.values()
                if e.get("task") == "forecasting" and e.get("status") == "validated"
            ]
            if not forecasting:
                raise ArtifactNotAvailableError(
                    "No validated forecasting model in the registry — run the "
                    "training pipeline first."
                )
            selected = next((e for e in forecasting if e.get("selected")), None)
            if selected is None:
                raise ArtifactNotAvailableError(
                    "No forecasting artifact marked selected=true in the registry."
                )

            artifact_path = selected["artifact_path"]
            if artifact_path is None:
                # trivial baselines (naive / ma) have no parameters file
                name = selected["model_name"]
                if "ma" in name:
                    model = _MovingAverageArtifact(window=int(selected["hyperparameters"].get("window", 7)))
                else:
                    model = _NaiveArtifact()
                seq_len = 14
                checksum_ok = True  # nothing to verify
            else:
                if not os.path.exists(artifact_path):
                    raise ArtifactNotAvailableError(
                        f"Forecasting artifact missing: {artifact_path}"
                    )
                if self._checksum(artifact_path) != selected["checksum"]:
                    raise ArtifactNotAvailableError(
                        f"Artifact checksum mismatch: {artifact_path} — refusing to use it."
                    )
                data = np.load(artifact_path)
                name = selected["model_name"]
                if name == "finmind_ar14":
                    model = _PooledARArtifact(data["intercept"], data["coef"], int(data["lags"]))
                else:
                    model = _RecurrentArtifact(
                        str(data["cell"]), int(data["hidden_size"]), int(data["seq_len"]),
                        data["Wx"], data["Uh"], data["b"], data["Wy"], data["by"],
                    )
                seq_len = int(data["seq_len"])
                checksum_ok = True

            self._loaded = {
                "name": name,
                "model": model,
                "seq_len": seq_len,
                "registry_entry": selected,
                "candidates": {
                    e["model_name"]: e["metrics"].get("pooled_recursive_30d_scaled")
                    for e in forecasting
                    if e.get("metrics")
                },
                "checksum_verified": checksum_ok,
            }
            return self._loaded


_manager = ForecastModelManager()


def get_forecast_manager() -> ForecastModelManager:
    return _manager


def reset_manager() -> None:
    """Test hook — force a reload from the registry on next use."""
    global _manager
    _manager = ForecastModelManager()


# ---------------------------------------------------------------------------
# Prediction (validation -> preprocessing -> model -> postprocessing)
# ---------------------------------------------------------------------------
def forecast_series(daily_values: List[float], horizon: int) -> Dict:
    """Forecast the next `horizon` steps of a business's daily series using
    the registry-selected artifact. No training happens here.

    Returns real, structured output with provenance.
    """
    manager = get_forecast_manager()
    loaded = manager.load()
    model, seq_len = loaded["model"], loaded["seq_len"]
    entry = loaded["registry_entry"]

    values = np.asarray(daily_values, dtype=np.float64).ravel()
    if len(values) < seq_len:
        raise ValueError(
            f"at least {seq_len} days of history are required; got {len(values)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("series contains non-finite values")
    horizon = int(max(1, min(horizon, 90)))

    # Inference-time preprocessing: z-score with the business's OWN observed
    # history (past only — the future is unknown, so no leakage is possible).
    mu = float(values.mean())
    sd = float(values.std())
    if sd == 0:
        raise ValueError("series has zero variance; a forecast is undefined")
    scaled = (values - mu) / sd

    predicted_scaled = model.forecast(scaled[-seq_len:], horizon)
    predicted = predicted_scaled * sd + mu

    return {
        "predictions": [round(float(v), 2) for v in predicted],
        "model": {
            "artifact": loaded["name"],
            "model_type": entry["model_type"],
            "version": entry.get("version"),
            "selection_method": (
                "offline walk-forward evaluation on held-out daily financial "
                "series; selected by pooled recursive-30-day scaled test MAE "
                "(see data/models/registry.json)"
            ),
            "artifact_checksum_verified": bool(loaded["checksum_verified"]),
            "trained_on_days": _artifact_training_days(entry),
            "training_datasets": entry.get("dataset"),
            "offline_holdout_metrics": entry.get("metrics", {}).get("pooled_recursive_30d_scaled"),
            "candidate_metrics": loaded["candidates"],
        },
        "preprocessing": {
            "method": "z-score with the business's own observed history",
            "series_mean": round(mu, 2),
            "series_std": round(sd, 2),
            "context_days": int(len(values)),
            "context_window": seq_len,
        },
    }


def _artifact_training_days(entry: Dict) -> int:
    """Real number of daily observations the artifact was trained on
    (recorded by the training pipeline in the registry entry)."""
    days = entry.get("training_days")
    if days is None:
        raise ArtifactNotAvailableError(
            "Registry entry lacks training_days — re-run the training pipeline."
        )
    return int(days)
