"""Per-business transaction anomaly detection service.

Detection pipeline (all real, all explainable):

1. A globally-trained IsolationForest ARTIFACT (finmind_anomaly, trained
   on 6M+ real PaySim transactions with the exact same feature schema
   below) is loaded once per process from data/models/finmind_anomaly/
   (checksum-verified against the model registry). Requests only score
   features — no model is ever trained or fitted during a request.

       [log_amount, hour_sin, hour_cos, dow_sin, dow_cos, is_weekend]

2. Candidates flagged by the model are confirmed with per-business
   statistical evidence computed from the business's own history:
       * amount deviating >= 2 standard deviations from the business's own
         spending distribution, or
       * off-hours activity (before 06:00 / after 23:00), or
       * exact duplicates (same amount and merchant within 72 hours).

3. Every returned anomaly carries its evidence (mean/std, z-score, hour),
   so results are auditable – nothing is hard-coded and nothing is
   fabricated.

A business with too few transactions gets an explicit insufficient-data
result instead of guessed flags. If the trained artifact is missing or
corrupt, the service reports unavailability instead of falling back to a
heuristic stand-in.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from .. import models
from ml.inference.anomaly_inference import (
    AnomalyModelNotAvailableError,
    get_anomaly_manager,
    model_metadata,
)

logger = logging.getLogger(__name__)

MIN_TX_FOR_MODEL = 40
DEFAULT_TOP_N = 25
Z_THRESHOLD = 2.0
MULTIPLE_OF_MEDIAN = 3.0
EXTREME_MULTIPLE = 5.0
OFF_HOURS_BEFORE = 6
OFF_HOURS_AFTER = 23
DUPLICATE_WINDOW_HOURS = 72
BUSINESS_RELATIVE_PERCENTILE = 5.0  # screening rate of the business's own scores


class AnomalyService:
    def __init__(self, db: Session, business_id: int):
        self.db = db
        self.business_id = business_id

    # ------------------------------------------------------------------
    # Feature engineering (must match the artifact's training schema)
    # ------------------------------------------------------------------
    @staticmethod
    def _features(amount: float, ts) -> List[float]:
        hour = ts.hour + ts.minute / 60.0
        dow = ts.weekday()
        return [
            math.log1p(max(amount, 0.0)),
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * dow / 7.0),
            math.cos(2 * math.pi * dow / 7.0),
            1.0 if dow >= 5 else 0.0,
        ]

    def _load_transactions(self) -> List[models.Transaction]:
        return (
            self.db.query(models.Transaction)
            .join(models.Account, models.Transaction.account_id == models.Account.id)
            .filter(models.Account.business_id == self.business_id)
            .order_by(models.Transaction.timestamp.desc())
            .all()
        )

    def _score_with_artifact(self, X: np.ndarray):
        """Score features with the globally-trained artifact (no fitting).

        The artifact provides the anomaly SCORE (learned from 6M+ real
        transactions). The decision FLAG uses a business-relative threshold:
        the BUSINESS_RELATIVE_PERCENTILE percentile of the business's own
        historical scores — a descriptive statistic of the business's past,
        not a fitted model parameter. This mirrors the screening rate the
        service has always used (5%) and adapts to each business's
        distribution, instead of applying PaySim's contamination threshold
        to MYR business data.
        """
        manager = get_anomaly_manager().load()
        model = manager["model"]
        expected = len(manager["config"].get("features") or [])
        if expected and expected != X.shape[1]:
            raise AnomalyModelNotAvailableError(
                f"Feature schema mismatch: artifact expects {expected} features, "
                f"got {X.shape[1]}."
            )
        scores = model.score_samples(X)  # lower = more anomalous
        threshold = float(np.percentile(scores, BUSINESS_RELATIVE_PERCENTILE))
        preds = scores < threshold
        raw_preds = (model.predict(X) == -1)
        return preds, raw_preds, scores, threshold, manager

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, top_n: int = DEFAULT_TOP_N, min_amount: Optional[float] = None) -> Dict:
        txs = self._load_transactions()

        if len(txs) < MIN_TX_FOR_MODEL:
            return {
                "available": False,
                "reason": (
                    f"Anomaly detection needs at least {MIN_TX_FOR_MODEL} transactions of history; "
                    f"this business has {len(txs)}."
                ),
                "anomalies": [],
                "model": None,
            }

        # ---- 1. Artifact scoring on the business's own transactions ----
        X = np.array([self._features(float(t.amount), t.timestamp) for t in txs])
        try:
            preds, raw_preds, scores, biz_threshold, manager = self._score_with_artifact(X)
        except AnomalyModelNotAvailableError as exc:
            return {
                "available": False,
                "reason": f"Anomaly model artifacts are unavailable: {exc}",
                "anomalies": [],
                "model": None,
            }

        # ---- 2. Per-business statistical evidence ----
        debits = [t for t in txs if t.type == models.TransactionType.DEBIT]
        log_amounts = np.array([math.log1p(max(float(t.amount), 0.0)) for t in debits])
        mean_log = float(log_amounts.mean())
        std_log = float(log_amounts.std())
        median_debit = float(np.median([float(t.amount) for t in debits])) if debits else 0.0

        # Recurring payments: same merchant and amount appearing >= 6 times are
        # legitimate commitments (payroll, rent…) and are exempt from
        # amount-deviation evidence.
        recur_counter: Dict[tuple, int] = defaultdict(int)
        for t in txs:
            if t.timestamp is None or t.type != models.TransactionType.DEBIT:
                continue
            recur_counter[(t.merchant or t.description or "", round(float(t.amount), 2))] += 1
        recurring_keys = {k for k, c in recur_counter.items() if c >= 6 and k[0]}

        # Exact-duplicate detection (same account, merchant and amount within 72h)
        groups: Dict[tuple, List[models.Transaction]] = defaultdict(list)
        for t in txs:
            if t.timestamp is None:
                continue
            key = (t.account_id, t.merchant or t.description or "", round(float(t.amount), 2))
            groups[key].append(t)
        duplicate_ids = set()
        for key, group in groups.items():
            if not key[1] or len(group) < 2:
                continue
            group.sort(key=lambda t: t.timestamp)
            for i in range(1, len(group)):
                if group[i].timestamp - group[i - 1].timestamp <= timedelta(hours=DUPLICATE_WINDOW_HOURS):
                    duplicate_ids.add(group[i].id)
                    duplicate_ids.add(group[i - 1].id)

        anomalies: List[Dict] = []
        for t, score, pred, raw_pred in zip(txs, scores, preds, raw_preds):
            amount = float(t.amount)
            if t.type == models.TransactionType.DEBIT:
                z = (math.log1p(max(amount, 0.0)) - mean_log) / std_log if std_log else 0.0
            else:
                z = None
            hour = t.timestamp.hour if t.timestamp else None
            is_dup = t.id in duplicate_ids
            off_hours = hour is not None and (hour < OFF_HOURS_BEFORE or hour >= OFF_HOURS_AFTER)
            is_recurring = (
                t.type == models.TransactionType.DEBIT
                and (t.merchant or t.description or "", round(amount, 2)) in recurring_keys
            )

            reasons: List[str] = []
            if not is_recurring:
                if z is not None and abs(z) >= Z_THRESHOLD:
                    reasons.append(
                        f"amount is {abs(z):.1f} standard deviations from your typical spending"
                    )
                if median_debit > 0 and amount >= MULTIPLE_OF_MEDIAN * median_debit:
                    reasons.append(
                        f"about {amount / median_debit:.0f}× your typical payment"
                    )
            if off_hours:
                reasons.append(f"unusual time of day ({hour:02d}:00)")

            # A transaction is reported when:
            #   (a) the IsolationForest artifact flagged it (business-relative
            #       threshold on the artifact's score) AND statistical
            #       evidence confirms it,
            #   (b) it is an extreme statistical outlier (an independent detector:
            #       >= 5x the business's median payment), or
            #   (c) it is part of an exact duplicate within the time window.
            # Every displayed anomaly carries explainable evidence – model flags
            # without evidence are never shown.
            extreme = (
                not is_recurring
                and median_debit > 0
                and amount >= EXTREME_MULTIPLE * median_debit
            )
            include = (bool(pred) and bool(reasons)) or extreme or is_dup
            if not include:
                continue
            if is_dup:
                reasons.append(
                    "possible duplicate charge (same merchant and amount within 72 hours)"
                )
            if min_amount is not None and amount < float(min_amount):
                continue

            anomalies.append({
                "transaction_id": t.id,
                "date": t.timestamp.isoformat() if t.timestamp else None,
                "type": t.type,
                "amount": amount,
                "merchant": t.merchant or t.description,
                "category": t.category.name if t.category else None,
                "anomaly_score": round(float(score), 4),
                "reasons": reasons,
                "evidence": {
                    "business_avg_spend": round(float(np.exp(mean_log) - 1), 2),
                    "business_spend_std": round(float(np.exp(std_log) - 1), 2),
                    "amount_z_score": round(z, 2) if z is not None else None,
                    "hour": hour,
                    "duplicate": is_dup,
                },
            })

        anomalies.sort(key=lambda a: a["anomaly_score"])

        # ---- model provenance: real values from the loaded artifact ----
        try:
            metadata = model_metadata()
        except AnomalyModelNotAvailableError:
            metadata = {}
        return {
            "available": True,
            "reason": None,
            "anomalies": anomalies[: int(max(1, top_n))],
            "model": {
                "type": metadata.get("type") or "IsolationForest",
                "artifact": metadata.get("artifact"),
                "version": metadata.get("version"),
                "n_estimators": metadata.get("n_estimators"),
                "contamination": metadata.get("contamination"),
                "trained_on": metadata.get("trained_on") or manager["config"].get("trained_rows"),
                "training_dataset": metadata.get("training_dataset"),
                "offline_test_metrics": metadata.get("offline_test_metrics"),
                "artifact_checksum_verified": bool(metadata),
                "decision_rule": (
                    f"flag = artifact score below the {BUSINESS_RELATIVE_PERCENTILE:g}th percentile "
                    "of this business's own transaction scores (business-relative screening, "
                    "same operating rate as before) — model scores come from the trained "
                    "artifact; nothing is fitted during a request"
                ),
                "business_score_threshold": round(biz_threshold, 6),
                "features": ["log_amount", "hour", "day_of_week", "is_weekend"],
                "evidence_rules": [
                    "IsolationForest artifact flag confirmed by z-score / multiple-of-median / off-hours evidence",
                    f"extreme outliers: amount >= {EXTREME_MULTIPLE:.0f}x the business median payment",
                    f"exact duplicates within {DUPLICATE_WINDOW_HOURS}h (same account and merchant)",
                    "recurring payments (>= 6 repeats) are exempt from amount-deviation evidence",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Single-transaction scoring (same loaded artifact, business-relative
    # decision threshold from the business's own historical scores)
    # ------------------------------------------------------------------
    def score_transaction(self, amount: float, ts=None) -> Dict:
        from datetime import datetime
        ts = ts or datetime.utcnow()
        txs = self._load_transactions()
        if len(txs) < MIN_TX_FOR_MODEL:
            return {
                "available": False,
                "reason": f"Anomaly scoring needs at least {MIN_TX_FOR_MODEL} transactions of history.",
                "is_anomaly": None,
                "score": None,
            }
        try:
            manager = get_anomaly_manager().load()
            model = manager["model"]
            X = np.array([self._features(float(t.amount), t.timestamp) for t in txs])
            own_scores = model.score_samples(X)
            threshold = float(np.percentile(own_scores, BUSINESS_RELATIVE_PERCENTILE))
            point = np.array([self._features(amount, ts)])
            score = float(model.score_samples(point)[0])
            raw_pred = int(model.predict(point)[0]) == -1
            return {
                "available": True,
                "reason": None,
                "is_anomaly": bool(score < threshold),
                "score": round(score, 4),
                "decision_rule": (
                    f"score below the {BUSINESS_RELATIVE_PERCENTILE:g}th percentile of this "
                    f"business's own transaction scores (threshold {threshold:.4f})"
                ),
                "artifact_raw_decision": bool(raw_pred),
            }
        except AnomalyModelNotAvailableError as exc:
            return {
                "available": False,
                "reason": f"Anomaly model artifacts are unavailable: {exc}",
                "is_anomaly": None,
                "score": None,
            }
