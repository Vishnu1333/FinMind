"""Artifact-backed anomaly scoring (Phase 10).

Loads the globally-trained IsolationForest artifact (finmind_anomaly,
trained on PaySim with the live feature schema) once per process and
verifies its checksum against the model registry. Requests only ever
score features — nothing is trained or fitted here.

Feature contract (must match backend AnomalyService._features):
    [log1p(amount), hour_sin, hour_cos, dow_sin, dow_cos, is_weekend]
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

import numpy as np

REGISTRY_PATH = "data/models/registry.json"


class AnomalyModelNotAvailableError(RuntimeError):
    pass


class AnomalyModelManager:
    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry_path = registry_path
        self._lock = threading.Lock()
        self._loaded: Optional[Dict] = None

    def _checksum(self, path: str) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def load(self) -> Dict:
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            if not os.path.exists(self.registry_path):
                raise AnomalyModelNotAvailableError(
                    "Model registry not found at data/models/registry.json — "
                    "run ml.training.train_fraud_models first."
                )
            with open(self.registry_path) as f:
                registry = json.load(f)
            entry = registry.get("finmind_anomaly")
            if not entry or entry.get("status") != "validated":
                raise AnomalyModelNotAvailableError(
                    "No validated anomaly model in the registry "
                    "(expected finmind_anomaly)."
                )
            artifact_path = entry["artifact_path"]
            if not os.path.exists(artifact_path):
                raise AnomalyModelNotAvailableError(
                    f"Anomaly artifact missing: {artifact_path}"
                )
            if self._checksum(artifact_path) != entry["checksum"]:
                raise AnomalyModelNotAvailableError(
                    f"Anomaly artifact checksum mismatch: {artifact_path}"
                )
            import joblib

            model = joblib.load(artifact_path)
            config_path = os.path.join(os.path.dirname(artifact_path), "config.json")
            with open(config_path) as f:
                config = json.load(f)
            self._loaded = {
                "model": model,
                "entry": entry,
                "config": config,
                "n_estimators": config.get("model_params", entry.get("hyperparameters", {})).get("n_estimators", 100),
                "contamination": entry.get("hyperparameters", {}).get("contamination"),
                "trained_on": config.get("trained_rows"),
            }
            return self._loaded


_manager = AnomalyModelManager()


def get_anomaly_manager() -> AnomalyModelManager:
    return _manager


def reset_manager() -> None:
    global _manager
    _manager = AnomalyModelManager()


def score_features(features: List[float]) -> Dict:
    """Score a single feature vector with the loaded artifact.

    Returns {'is_anomaly': bool, 'score': float} where score is the
    IsolationForest decision score (lower = more anomalous) — the raw
    model output, never invented.
    """
    loaded = get_anomaly_manager().load()
    model = loaded["model"]
    point = np.asarray(features, dtype=np.float64).reshape(1, -1)
    pred = int(model.predict(point)[0])
    score = float(model.score_samples(point)[0])
    return {"is_anomaly": pred == -1, "score": round(score, 4)}


def score_batch(features: List[List[float]]) -> Dict:
    """Score many feature vectors; returns raw model outputs."""
    loaded = get_anomaly_manager().load()
    model = loaded["model"]
    points = np.asarray(features, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    preds = (model.predict(points) == -1).astype(int)
    scores = model.score_samples(points)
    return {"predictions": preds.tolist(), "scores": [round(float(s), 4) for s in scores]}


def model_metadata() -> Dict:
    """Real provenance for API/tool payloads."""
    loaded = get_anomaly_manager().load()
    entry, config = loaded["entry"], loaded["config"]
    return {
        "type": entry.get("model_type"),
        "artifact": "finmind_anomaly",
        "version": entry.get("version"),
        "n_estimators": loaded["n_estimators"],
        "contamination": loaded["contamination"],
        "trained_on": loaded["trained_on"],
        "training_dataset": entry.get("dataset"),
        "offline_test_metrics": entry.get("metrics"),
        "features": config.get("features"),
        "artifact_checksum_verified": True,
    }
