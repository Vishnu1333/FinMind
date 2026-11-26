"""Per-business cash-flow forecasting service.

Pipeline (Phase 10/11 — artifact-backed inference, NO training at request
time):

    financial data (authoritative PostgreSQL transactions)
        -> validation (>= MIN_TRAIN_DAYS of history required)
        -> preprocessing (daily aggregation + z-score with the business's
           own observed history)
        -> registry-selected model artifact loaded from
           data/models/finmind_<rnn|lstm|gru|ar14|...>/ (checksum-verified,
           loaded once per process)
        -> recursive multi-step inference
        -> inverse transform + prediction payload with honest provenance

The selected artifact and every reported offline metric come from the
training pipeline (ml.training.train_forecast_models) and the model
registry (data/models/registry.json). Nothing here is hard-coded or
invented: if the business has insufficient history the service reports
that explicitly, and if no trained artifact exists it says so instead of
falling back to heuristics.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models
from ml.inference.forecast_inference import (
    ArtifactNotAvailableError,
    forecast_series,
    get_forecast_manager,
)

logger = logging.getLogger(__name__)

HORIZON_CAP = 90
MIN_TRAIN_DAYS = 60  # minimum business history for a reliable forecast


class InsufficientDataError(RuntimeError):
    """Raised when the business does not have enough history to forecast."""


class ForecastService:
    """Serves forecasts for a business from the trained global artifact."""

    def __init__(self, db: Session, business_id: int):
        self.db = db
        self.business_id = business_id

    # ------------------------------------------------------------------
    # Data preparation (unchanged: authoritative daily net cash-flow series)
    # ------------------------------------------------------------------
    def _daily_net_series(self) -> List[Dict]:
        """Aggregate real transactions into a daily net cash-flow series."""
        rows = (
            self.db.query(
                func.date(models.Transaction.timestamp).label("day"),
                models.Transaction.type,
                func.sum(models.Transaction.amount).label("total"),
            )
            .join(models.Account, models.Transaction.account_id == models.Account.id)
            .filter(models.Account.business_id == self.business_id)
            .group_by("day", models.Transaction.type)
            .order_by("day")
            .all()
        )
        by_day: Dict[str, float] = {}
        for day, typ, total in rows:
            key = str(day)
            value = float(total or 0)
            by_day[key] = by_day.get(key, 0.0) + (value if typ == models.TransactionType.CREDIT else -value)
        if not by_day:
            return []
        # Fill gaps with 0.0 net flow (no activity that day).
        first = min(by_day.keys())
        last = max(by_day.keys())
        series = []
        d = datetime.strptime(str(first), "%Y-%m-%d").date()
        end = datetime.strptime(str(last), "%Y-%m-%d").date()
        while d <= end:
            series.append({"date": d.isoformat(), "net": by_day.get(d.isoformat(), 0.0)})
            d += timedelta(days=1)
        return series

    # ------------------------------------------------------------------
    # Public API (artifact-backed; identical contract to the previous version)
    # ------------------------------------------------------------------
    def forecast(self, days: int = 30) -> Dict:
        days = int(max(1, min(days, HORIZON_CAP)))
        series = self._daily_net_series()
        if len(series) < MIN_TRAIN_DAYS:
            raise InsufficientDataError(
                f"At least {MIN_TRAIN_DAYS} days of transaction history are required for a "
                f"reliable forecast; found {len(series)}."
            )

        values = [p["net"] for p in series]
        try:
            result = forecast_series(values, days)  # loads artifact, no training
        except ValueError as exc:
            raise InsufficientDataError(str(exc))
        except ArtifactNotAvailableError as exc:
            raise InsufficientDataError(
                f"Forecast model artifacts are unavailable: {exc}"
            )

        artifact_model = result["model"]
        last_date = datetime.strptime(series[-1]["date"], "%Y-%m-%d").date()
        forecast = [
            {
                "day": i + 1,
                "date": (last_date + timedelta(days=i + 1)).isoformat(),
                "predicted_cashflow": p,
            }
            for i, p in enumerate(result["predictions"])
        ]
        cumulative = 0.0
        for p in forecast:
            cumulative += p["predicted_cashflow"]
            p["cumulative"] = round(cumulative, 2)

        holdout = artifact_model.get("offline_holdout_metrics") or {}
        candidates = artifact_model.get("candidate_metrics") or {}

        return {
            "currency": "MYR",
            "forecast": forecast,
            "model": {
                "selected": artifact_model["model_type"],
                "selection_method": artifact_model["selection_method"],
                "artifact": artifact_model["artifact"],
                "version": artifact_model["version"],
                "artifact_checksum_verified": artifact_model["artifact_checksum_verified"],
                "trained_on_days": artifact_model["trained_on_days"],
                "training_datasets": artifact_model["training_datasets"],
                "business_context_days": result["preprocessing"]["context_days"],
                "context_window": result["preprocessing"]["context_window"],
                "holdout_days": holdout.get("n"),
                "holdout_mae_scaled": round(holdout.get("mae"), 4) if holdout.get("mae") is not None else None,
                "holdout_rmse_scaled": round(holdout.get("rmse"), 4) if holdout.get("rmse") is not None else None,
                "candidates": {
                    name: {
                        "holdout_mae_scaled": round(m["mae"], 4),
                        "holdout_rmse_scaled": round(m["rmse"], 4),
                    }
                    for name, m in candidates.items()
                    if m
                },
                "series_mean": result["preprocessing"]["series_mean"],
                "series_std": result["preprocessing"]["series_std"],
            },
        }


# ---------------------------------------------------------------------------
# Process-level cache: a request only performs forward passes, but the DB
# aggregation is still worth caching per business until new transactions
# arrive (cache key = business + days + tx count).
# ---------------------------------------------------------------------------
_cache: Dict[tuple, Dict] = {}
_lock = threading.Lock()


def get_cached_forecast(db: Session, business_id: int, days: int) -> Dict:
    """Return the forecast for a business, loading the trained artifact
    (never training) and recomputing only when the underlying transaction
    data has changed (cache key = business + days + tx count)."""
    tx_count = (
        db.query(func.count(models.Transaction.id))
        .join(models.Account, models.Transaction.account_id == models.Account.id)
        .filter(models.Account.business_id == business_id)
        .scalar()
    )
    key = (business_id, int(days), int(tx_count or 0))
    with _lock:
        if key in _cache:
            return _cache[key]

    service = ForecastService(db, business_id)
    result = service.forecast(days=days)
    with _lock:
        # Bounded cache: keep at most 64 business entries.
        if len(_cache) > 64:
            _cache.clear()
        _cache[key] = result
    return result


def available_model_info() -> Optional[Dict]:
    """Registry summary of the forecasting artifacts (for health checks)."""
    try:
        loaded = get_forecast_manager().load()
        return {
            "artifact": loaded["name"],
            "model_type": loaded["registry_entry"]["model_type"],
            "checksum_verified": loaded["checksum_verified"],
            "candidates": list(loaded["candidates"].keys()),
        }
    except ArtifactNotAvailableError as exc:
        return {"available": False, "reason": str(exc)}
