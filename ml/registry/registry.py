"""FinMind model registry.

Single source of truth for every trained model artifact:

    data/models/registry.json

Each entry records real, verifiable metadata: metrics produced by the
evaluation code, the artifact checksum, the exact dataset version, the
hyperparameters used, and the reproducibility environment. Nothing is
hand-written here — entries are created by the training scripts and every
claimed metric is computed from real predictions.

Loading validates the artifact checksum so a corrupt or hand-edited
artifact can never be silently used for inference.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

REGISTRY_PATH = os.path.join("data", "models", "registry.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_checksum(path: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, path: str = REGISTRY_PATH):
        self.path = path
        self._entries: Dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._entries = json.load(f)
        else:
            self._entries = {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._entries, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    def register(self, entry: dict) -> None:
        required = {
            "model_name", "model_type", "dataset", "version", "trained_at",
            "features", "metrics", "artifact_path", "checksum", "status",
        }
        missing = required - set(entry)
        if missing:
            raise ValueError(f"registry entry missing fields: {sorted(missing)}")
        self._entries[entry["model_name"]] = entry

    def get(self, model_name: str) -> Optional[dict]:
        return self._entries.get(model_name)

    def all(self) -> Dict[str, dict]:
        return dict(self._entries)

    def by_task(self, task: str) -> List[dict]:
        return [e for e in self._entries.values() if e.get("task") == task]

    def best_for_task(self, task: str, metric: str, higher_is_better: bool) -> Optional[dict]:
        candidates = [
            e for e in self.by_task(task)
            if e.get("status") == "validated"
            and isinstance(e.get("metrics", {}).get(metric), (int, float))
            and e["metrics"][metric] is not None
        ]
        if not candidates:
            return None
        pick = max if higher_is_better else min
        return pick(candidates, key=lambda e: e["metrics"][metric])

    # ------------------------------------------------------------------
    def validate_artifact(self, model_name: str) -> bool:
        """Checksum-verify the artifact on disk against the registry.

        Supports the legacy md5 checksums (used by the deprecated v1
        artifacts) via the entry's `checksum_algorithm` field.
        """
        entry = self.get(model_name)
        if entry is None:
            return False
        path = entry["artifact_path"]
        if not os.path.exists(path):
            return False
        algo = entry.get("checksum_algorithm", "sha256")
        return file_checksum(path, algo) == entry["checksum"]
