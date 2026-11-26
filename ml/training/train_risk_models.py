"""Credit-risk model training: UCI Default + Statlog German Credit.

Real supervised training on real labelled datasets:

* finmind_risk_credit  — Default of Credit Card Clients (Taiwan, 30k rows)
* finmind_risk_german  — Statlog German Credit (1k rows)

Both datasets are client-snapshot tabular data (NO timestamps), so a
stratified random split is valid here and is documented as such
(time-series rules in ml.preprocessing.timeseries do not apply).

Protocol:
* 70/15/15 train/val/test, stratified, seed 42
* StandardScaler fitted on TRAIN only
* two candidates per dataset (LogisticRegression vs gradient boosting);
  the winner is chosen by VALIDATION PR-AUC; test metrics reported for both
* F1 threshold tuned on validation, never on test

These models are validated artifacts (real training, real metrics) for
credit-risk scoring. They are NOT wired to the live FinMind API because
FinMind businesses have no credit-feature schema; the inference module can
still load and run them (see backend tests).

Run:  ./backend/venv/bin/python -m ml.training.train_risk_models
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib

from ml.evaluation.metrics import best_f1_threshold, classification_metrics
from ml.registry.registry import ModelRegistry, file_checksum, utc_now_iso
from ml.training.common import environment_snapshot, write_json

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def stratified_split(y: np.ndarray):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_ids, temp = train_test_split(idx, test_size=1 - TRAIN_FRAC, random_state=SEED, stratify=y)
    relative_val = VAL_FRAC / (1 - TRAIN_FRAC)
    val_ids, test_ids = train_test_split(
        temp, test_size=1 - relative_val, random_state=SEED, stratify=y[temp]
    )
    return train_ids, val_ids, test_ids


def _candidates():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    return [
        ("logistic", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
        ("hgb", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                                class_weight="balanced", random_state=SEED)),
    ]


def train_one(name: str, npz_path: str, dataset: str, registry: ModelRegistry, env: dict) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int8)
    features = [str(f) for f in data["feature_names"]]
    print(f"  {name}: rows={len(X)} positives={int(y.sum())} ({y.mean():.4f})")

    train_ids, val_ids, test_ids = stratified_split(y)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X[train_ids])
    Xtr, Xva, Xte = scaler.transform(X[train_ids]), scaler.transform(X[val_ids]), scaler.transform(X[test_ids])
    ytr, yva, yte = y[train_ids], y[val_ids], y[test_ids]

    outcomes = {}
    fitted = {}
    for cand_name, model in _candidates():
        model.fit(Xtr, ytr)
        val_scores = model.predict_proba(Xva)[:, 1]
        val_metrics = classification_metrics(yva, val_scores, threshold=0.5)
        thr, val_f1 = best_f1_threshold(yva, val_scores)
        test_scores = model.predict_proba(Xte)[:, 1]
        test_metrics = classification_metrics(yte, test_scores, threshold=thr)
        outcomes[cand_name] = {
            "model": type(model).__name__,
            "val_metrics_at_0_5": val_metrics,
            "validation_tuned_threshold": thr,
            "validation_f1_at_threshold": val_f1,
            "test_metrics": test_metrics,
        }
        fitted[cand_name] = model
        print(f"    {cand_name}: test f1={test_metrics['f1']} pr_auc={test_metrics['pr_auc']} "
              f"roc_auc={test_metrics['roc_auc']}")

    # selection by VALIDATION PR-AUC (never test)
    best_cand = max(outcomes, key=lambda c: outcomes[c]["val_metrics_at_0_5"]["pr_auc"] or -1)
    print(f"    selected: {best_cand} (by validation PR-AUC)")

    model_dir = f"data/models/{name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(fitted[best_cand], model_path)
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    write_json(os.path.join(model_dir, "config.json"), {
        "model": outcomes[best_cand]["model"],
        "seed": SEED, "features": features,
        "split": f"stratified random {TRAIN_FRAC}/{VAL_FRAC}/{round(1-TRAIN_FRAC-VAL_FRAC,2)} (no timestamps in data)",
        "scaler": "StandardScaler fit on train only",
        "trained_rows": int(len(train_ids)),
        "trained_at": utc_now_iso(),
    })
    write_json(os.path.join(model_dir, "metrics.json"),
               {"selected": best_cand, "candidates": outcomes})

    registry.register({
        "model_name": name,
        "model_type": outcomes[best_cand]["model"],
        "task": "credit_risk_classification",
        "dataset": dataset,
        "version": "1.0",
        "trained_at": utc_now_iso(),
        "features": features,
        "target": data.get("target_name", "default") if hasattr(data, "get") else "default",
        "hyperparameters": {"random_state": SEED, "class_weight": "balanced",
                             "selection": "validation PR-AUC"},
        "metrics": outcomes[best_cand]["test_metrics"],
        "artifact_path": model_path,
        "checksum": file_checksum(model_path),
        "status": "validated",
        "serving": "artifact_only (no live credit-feature schema in FinMind)",
        "all_candidates_test_metrics": {c: o["test_metrics"] for c, o in outcomes.items()},
        "environment": env,
    })

    return {
        name: {
            "dataset": dataset,
            "rows": int(len(X)),
            "split": {"train": int(len(train_ids)), "val": int(len(val_ids)), "test": int(len(test_ids))},
            "candidates": outcomes,
            "selected": best_cand,
        }
    }


def main() -> None:
    print("=" * 70)
    print("FINMIND CREDIT-RISK MODEL TRAINING")
    print("=" * 70)
    env = environment_snapshot()
    registry = ModelRegistry()
    report = {
        "task": "credit_risk",
        "trained_at": utc_now_iso(),
        "random_seed": SEED,
        "environment": env,
        "split_rule": "stratified random (tabular snapshots without timestamps)",
    }
    report.update(train_one(
        "finmind_risk_credit",
        "data/processed/risk/uci_credit_default.npz",
        "Default of Credit Card Clients (uciml/default-of-credit-card-clients-dataset)",
        registry, env,
    ))
    report.update(train_one(
        "finmind_risk_german",
        "data/processed/risk/german_credit.npz",
        "Statlog German Credit Data (UCI 144)",
        registry, env,
    ))
    registry.save()
    write_json("data/reports/training/risk_training_report.json", report)
    print("Artifacts: data/models/finmind_{risk_credit,risk_german}")


if __name__ == "__main__":
    main()
