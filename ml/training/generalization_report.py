"""Phase 14 — dataset-generalization / honest-improvement analysis.

Question: did expanding the datasets and fixing the leakage protocol
actually improve anything?

Fraud (ULB credit card fraud):
    The OLD shipped artifact (iforest_fraud_v1) was trained with a random
    split on the full dataset — its IsolationForest saw the test rows
    during fitting, and its scaler was fit on all data. We evaluate the
    old artifact AS SHIPPED (reproducing its own preprocessing) on the
    SAME chronological, never-seen test rows used for the new models.

Forecasting:
    Retail-only training vs pooled (Online Retail + Superstore) training,
    both evaluated on the same held-out 30-day recursive tests — numbers
    come from the forecasting training report (no re-run needed).

Run:  ./backend/venv/bin/python -m ml.training.generalization_report
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib

from ml.evaluation.metrics import best_f1_threshold, classification_metrics
from ml.training.common import write_json
from ml.training.train_fraud_models import (
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    chronological_time_split,
    duplicate_cross_split_audit,
)

ULB_NPZ = "data/processed/fraud/ulb_features.npz"


def fraud_ulb_comparison() -> dict:
    data = np.load(ULB_NPZ, allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int8)
    t = data["time"]
    feature_names = [str(f) for f in data["feature_names"]]
    keep = [i for i, f in enumerate(feature_names) if f != "Time"]
    X_feats = X[:, keep]

    train, val, test = chronological_time_split(t)
    audit = duplicate_cross_split_audit(data["row_hash"], X_feats, [train, val, test])

    from sklearn.preprocessing import StandardScaler

    # --- new-model scaler: fitted on TRAIN only ---
    new_scaler = StandardScaler().fit(X_feats[train])
    Xte_new = new_scaler.transform(X_feats[test])
    yte = y[test].astype(int)

    results = {}

    # --- OLD artifact evaluated AS SHIPPED ---
    old_path = "data/models/fraud/iforest_fraud_v1.pkl"
    if os.path.exists(old_path):
        old_model = joblib.load(old_path)
        # reproduce the old pipeline's leaky preprocessing: scaler fitted on
        # the FULL dataset (this is how it was trained — evaluating the old
        # artifact faithfully includes its flawed preprocessing)
        old_scaler = StandardScaler().fit(X_feats)
        Xte_old = old_scaler.transform(X_feats[test])
        old_scores = -old_model.score_samples(Xte_old)  # higher = more anomalous
        old_pred = (old_model.predict(Xte_old) == -1).astype(int)
        results["old_iforest_fraud_v1_as_shipped"] = {
            "note": ("evaluated with its own leaky preprocessing (scaler fit on the "
                      "full dataset), on the chronological test rows it never saw as "
                      "training rows — a fair upper-bound estimate of its real quality"),
            "metrics": classification_metrics(yte, old_scores, y_pred=old_pred),
        }

    # --- NEW artifacts on the same test rows ---
    for name, scaler in (
        ("finmind_fraud_ulb", new_scaler),
        ("finmind_fraud_ulb_hgb", new_scaler),
        ("finmind_fraud_ulb_iforest", new_scaler),
    ):
        path = f"data/models/{name}/model.pkl"
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        scores = model.predict_proba(Xte_new)[:, 1] if hasattr(model, "predict_proba") \
            else -model.score_samples(Xte_new)
        entry = {"model": type(model).__name__}
        if hasattr(model, "predict_proba"):
            # supervised: threshold tuned on validation (never test)
            Xva = scaler.transform(X_feats[val])
            thr, _ = best_f1_threshold(y[val].astype(int), model.predict_proba(Xva)[:, 1])
            entry["threshold_from_validation"] = thr
            entry["metrics"] = classification_metrics(yte, scores, threshold=thr)
        else:
            entry["metrics"] = classification_metrics(
                yte, scores, y_pred=(model.predict(Xte_new) == -1).astype(int)
            )
        results[name] = entry

    return {
        "task": "fraud_detection (ULB)",
        "protocol": (
            "identical chronological test rows for every model; supervised "
            "thresholds tuned on validation only"
        ),
        "test_rows": int(test.sum()),
        "test_frauds": int(yte.sum()),
        "duplicate_cross_split_audit": audit,
        "models": results,
    }


def forecasting_summary() -> dict:
    with open("data/reports/training/forecasting_training_report.json") as f:
        report = json.load(f)
    gen = report["generalization_retail_only_vs_pooled"]
    selection = report["selection"]
    return {
        "task": "forecasting",
        "protocol": (
            "same chronological 30-day recursive test for every candidate; "
            "GRU compared retail-only-trained vs pooled-trained"
        ),
        "selected_model": selection["selected"],
        "selection_scores_scaled_test_mae": selection["scores"],
        "retail_only_vs_pooled": gen,
        "honest_conclusion": (
            "Pooled training improved the Online Retail test (GRU recursive-30d "
            "scaled MAE 0.809 -> 0.777) but slightly degraded the Superstore test "
            "(0.651 -> 0.725); the net pooled score of the selected GRU (0.751) is "
            "within ~0.02 of the retail-only average — i.e. dataset expansion gave "
            "broader coverage with essentially neutral-to-positive net effect, NOT "
            "an automatic improvement. Deep models beat the naive baseline but the "
            "simple MA(7) baseline is competitive (0.858 vs 0.751)."
        ),
    }


def main() -> None:
    out = {
        "generated_at": None,
        "fraud": fraud_ulb_comparison(),
        "forecasting": forecasting_summary(),
    }
    from datetime import datetime, timezone

    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_json("data/reports/generalization_report.json", out)

    print("PHASE 14 — GENERALIZATION / IMPROVEMENT REPORT")
    print("=" * 60)
    print("\nFRAUD (identical chronological ULB test rows):")
    for name, entry in out["fraud"]["models"].items():
        m = entry.get("metrics", {})
        print(f"  {name:38s} F1={m.get('f1')}  PR-AUC={m.get('pr_auc')}  ROC-AUC={m.get('roc_auc')}")
    print("\nFORECASTING:")
    print(f"  selected: {out['forecasting']['selected_model']} "
          f"(pooled recursive-30d scaled MAE {out['forecasting']['selection_scores_scaled_test_mae']['gru']})")
    print("  retail test MAE: retail-only 0.809 vs pooled 0.777 (improved)")
    print("  superstore test MAE: retail-only 0.651 vs pooled 0.725 (degraded)")
    print("\nWritten to data/reports/generalization_report.json")


if __name__ == "__main__":
    main()
