"""Shared helpers for all training scripts (artifacts, env capture, registry)."""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Dict

import numpy as np


def environment_snapshot() -> Dict[str, str]:
    """Reproducibility environment (Phase 16)."""
    import sklearn
    import statsmodels

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": __import__("pandas").__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "scipy": __import__("scipy").__version__,
    }


def save_recurrent_artifact(
    model_dir: str,
    model,
    config: Dict,
    metrics: Dict,
) -> Dict[str, str]:
    """Persist a trained RecurrentForecaster (weights + config + metrics).

    Returns the paths written. The .npz stores raw parameter arrays so
    inference never needs to re-import the training class.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.npz")
    np.savez(
        model_path,
        Wx=model.Wx, Uh=model.Uh, b=model.b, Wy=model.Wy, by=model.by,
        cell=model.cell, hidden_size=np.int64(model.H), seq_len=np.int64(model.seq_len),
    )
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return {
        "model": model_path,
        "config": config_path,
        "metrics": metrics_path,
    }


def sha256_of(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
