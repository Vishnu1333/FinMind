"""Fraud / anomaly model training: PaySim + ULB Credit Card Fraud.

Models trained (all REAL training on REAL data):

1. finmind_anomaly          — IsolationForest on PaySim train split using the
                              LIVE feature schema used by
                              backend/app/services/anomaly_service.py
                              [log_amount, hour_sin, hour_cos, dow_sin,
                               dow_cos, is_weekend].
                              This replaces the old iforest_fraud_v1 which
                              was (a) trained on PCA features that don't
                              exist for business transactions and (b) fitted
                              on data that included its own test rows.
2. finmind_fraud_paysim     — supervised gradient boosting classifier on
                              PaySim (labels: isFraud).
3. finmind_fraud_ulb        — supervised logistic classifier on ULB
                              (labels: Class), leakage-safe chronological
                              protocol.
4. finmind_fraud_ulb_iforest— unsupervised IsolationForest on ULB fitted on
                              TRAIN NORMALS ONLY (evaluation artifact).

Leakage protocol (Phase 15):
* PaySim: chronological split on `step` (1 step = 1 hour); post-transaction
  balances and isFlaggedFraud are excluded (dataset-author leakage warning).
* ULB: chronological split on Time; Time excluded as a feature (position
  leakage); scaler fitted on train only.
* Thresholds for F1 are tuned on VALIDATION, never on test.
* Duplicate-feature rows are audited for cross-split leakage.

Run:  ./backend/venv/bin/python -m ml.training.train_fraud_models
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

PAYSIM_NPZ = "data/processed/fraud/paysim_features.npz"
ULB_NPZ = "data/processed/fraud/ulb_features.npz"
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SEED = 42


def chronological_step_split(step: np.ndarray):
    """Split by step boundaries: train steps < t1 <= val steps < t2 <= test."""
    steps = np.unique(step)
    n = len(steps)
    t1 = steps[int(n * TRAIN_FRAC)]
    t2 = steps[int(n * (TRAIN_FRAC + VAL_FRAC))]
    train = step < t1
    val = (step >= t1) & (step < t2)
    test = step >= t2
    return train, val, test, {"train_steps": int(train.sum()), "val_steps": int(val.sum()),
                               "test_steps": int(test.sum()), "boundary_1": int(t1),
                               "boundary_2": int(t2)}


def chronological_time_split(t: np.ndarray):
    """Chronological split on exact Time BOUNDARY values: all rows sharing
    the same Time value land in the SAME split, so duplicate records can
    never straddle a boundary."""
    boundaries = np.sort(np.unique(t))
    n = len(boundaries)
    t1 = boundaries[int(n * TRAIN_FRAC)]
    t2 = boundaries[int(n * (TRAIN_FRAC + VAL_FRAC))]
    train = t < t1
    val = (t >= t1) & (t < t2)
    test = t >= t2
    return train, val, test


def duplicate_cross_split_audit(row_hash: np.ndarray, X: np.ndarray, masks) -> dict:
    """Two-level duplicate/leakage audit.

    Level 1 (row identity — the leakage criterion): groups of identical RAW
    records that appear in more than one split. A non-zero count here means
    the same recorded transaction was used for both training and testing.

    Level 2 (feature collision — informational): rows with identical feature
    vectors but different raw identities (distinct transactions that merely
    look alike). Reported but not leakage.
    """
    import pandas as pd

    split_id = np.zeros(len(row_hash), dtype=np.int8)
    for i, mask in enumerate(masks):
        split_id[mask] = i + 1

    rows = pd.DataFrame({"h": row_hash, "s": split_id})
    spanning = rows.groupby("h")["s"].nunique()
    n_row_dup = int(rows["h"].duplicated(keep=False).sum())
    n_row_spanning = int((spanning > 1).sum())

    feat_hash = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    frows = pd.DataFrame({"h": feat_hash, "s": split_id})
    fspanning = frows.groupby("h")["s"].nunique()
    n_feat_dup = int(frows["h"].duplicated(keep=False).sum())
    n_feat_spanning = int((fspanning > 1).sum())

    return {
        "row_identity_duplicates_total": n_row_dup,
        "row_identity_groups_spanning_multiple_splits": n_row_spanning,
        "feature_vector_collisions_total": n_feat_dup,
        "feature_collision_groups_spanning_multiple_splits": n_feat_spanning,
        "interpretation": (
            "row identity = full raw record (leakage if it spans splits); "
            "feature collisions = distinct transactions with identical "
            "feature vectors (informational, not leakage)"
        ),
        "passed": n_row_spanning == 0,
    }


# ---------------------------------------------------------------------------
# PaySim
# ---------------------------------------------------------------------------
def train_paysim(registry: ModelRegistry, env: dict) -> dict:
    print("  loading PaySim features ...", end=" ", flush=True)
    data = np.load(PAYSIM_NPZ, allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int8)
    step = data["step"]
    X_anom = data["X_anomaly_schema"].astype(np.float64)
    anomaly_features = [str(f) for f in data["anomaly_feature_names"]]
    row_hash = data["row_hash"]
    print(f"rows={len(X)} fraud_rate={y.mean():.5f}")

    train, val, test, sizes = chronological_step_split(step)
    print(f"    chronological split by step: {sizes}")

    dup_audit = duplicate_cross_split_audit(row_hash, X, [train, val, test])
    print(
        f"    row-identity cross-split audit: {'PASS' if dup_audit['passed'] else 'FAIL'} "
        f"(spanning raw-record groups: {dup_audit['row_identity_groups_spanning_multiple_splits']}); "
        f"feature collisions across splits: {dup_audit['feature_collision_groups_spanning_multiple_splits']} (informational)"
    )

    results = {}

    # ---------------------------------------------------------------
    # 1. Live-schema IsolationForest (finmind_anomaly)
    # ---------------------------------------------------------------
    from sklearn.ensemble import IsolationForest

    train_fraud_rate = float(y[train].mean())
    iforest = IsolationForest(
        contamination=train_fraud_rate, random_state=SEED, n_estimators=100
    )
    iforest.fit(X_anom[train])
    # AUC on the test split with natural prevalence (labels are real)
    test_scores = -iforest.score_samples(X_anom[test])  # higher = more anomalous
    if_test_pred = (iforest.predict(X_anom[test]) == -1).astype(int)
    if_metrics = classification_metrics(
        y[test].astype(int), test_scores, y_pred=if_test_pred
    )
    results["finmind_anomaly"] = {
        "model": "IsolationForest",
        "trained_on_rows": int(train.sum()),
        "contamination": train_fraud_rate,
        "features": anomaly_features,
        "test_metrics": if_metrics,
    }
    print(f"    finmind_anomaly (IF, live schema) test: {if_metrics}")

    # ---------------------------------------------------------------
    # 2. Supervised classifier (finmind_fraud_paysim)
    # ---------------------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.1, max_depth=8, random_state=SEED,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
    )
    clf.fit(X[train], y[train])
    val_scores = clf.predict_proba(X[val])[:, 1]
    thr, val_f1 = best_f1_threshold(y[val].astype(int), val_scores)
    test_scores_c = clf.predict_proba(X[test])[:, 1]
    clf_metrics = classification_metrics(
        y[test].astype(int), test_scores_c, threshold=thr
    )
    results["finmind_fraud_paysim"] = {
        "model": "HistGradientBoostingClassifier",
        "trained_on_rows": int(train.sum()),
        "test_metrics": clf_metrics,
        "validation_tuned_threshold": thr,
        "validation_f1_at_threshold": val_f1,
    }
    print(f"    finmind_fraud_paysim (HGB) test: f1={clf_metrics['f1']} pr_auc={clf_metrics['pr_auc']}")

    # ---- save artifacts + registry ----
    for name, payload, model_obj, extra in [
        ("finmind_anomaly", results["finmind_anomaly"], iforest,
         {"features": anomaly_features, "trained_on": int(train.sum())}),
        ("finmind_fraud_paysim", results["finmind_fraud_paysim"], clf,
         {"features": [str(f) for f in data["feature_names"]], "trained_on": int(train.sum())}),
    ]:
        model_dir = f"data/models/{name}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model_obj, model_path)
        write_json(os.path.join(model_dir, "config.json"), {
            "model": payload["model"], "seed": SEED,
            "features": extra["features"], "target": "isFraud",
            "split": "chronological on step (70/15/15)",
            "trained_rows": extra["trained_on"],
            "trained_at": utc_now_iso(),
        })
        write_json(os.path.join(model_dir, "metrics.json"), payload)
        registry.register({
            "model_name": name,
            "model_type": payload["model"],
            "task": "fraud_detection" if "fraud" in name else "anomaly_detection",
            "dataset": "PaySim synthetic mobile money transactions (ealaxi/paysim1)",
            "version": "1.0",
            "trained_at": utc_now_iso(),
            "features": extra["features"],
            "target": "isFraud",
            "hyperparameters": (
                {"n_estimators": 100, "contamination": train_fraud_rate, "random_state": SEED}
                if name == "finmind_anomaly"
                else {"max_iter": 120, "learning_rate": 0.1, "max_depth": 8,
                      "class_weight": "balanced", "random_state": SEED}
            ),
            "metrics": payload["test_metrics"],
            "artifact_path": model_path,
            "checksum": file_checksum(model_path),
            "status": "validated",
            "leakage_audit": dup_audit,
            "environment": env,
        })

    return {
        "paysim": {
            "split": sizes,
            "duplicate_cross_split_audit": dup_audit,
            "models": results,
        }
    }


# ---------------------------------------------------------------------------
# ULB credit card fraud
# ---------------------------------------------------------------------------
def train_ulb(registry: ModelRegistry, env: dict) -> dict:
    print("  loading ULB features ...", end=" ", flush=True)
    data = np.load(ULB_NPZ, allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int8)
    t = data["time"]
    feature_names = [str(f) for f in data["feature_names"]]
    # Time is EXCLUDED as a feature (documented leakage decision)
    keep = [i for i, f in enumerate(feature_names) if f != "Time"]
    X_feats = X[:, keep]
    used_features = [feature_names[i] for i in keep]
    print(f"rows={len(X)} fraud_rate={y.mean():.5f} (Time column excluded as feature)")

    train, val, test = chronological_time_split(t)
    dup_audit = duplicate_cross_split_audit(data["row_hash"], X_feats, [train, val, test])
    print(
        f"    row-identity cross-split audit: {'PASS' if dup_audit['passed'] else 'FAIL'} "
        f"(spanning raw-record groups: {dup_audit['row_identity_groups_spanning_multiple_splits']}); "
        f"feature collisions across splits: {dup_audit['feature_collision_groups_spanning_multiple_splits']} (informational)"
    )

    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # scaler fitted on TRAIN only (leakage fix vs the old pipeline)
    scaler = StandardScaler().fit(X_feats[train])
    Xtr, Xva, Xte = scaler.transform(X_feats[train]), scaler.transform(X_feats[val]), scaler.transform(X_feats[test])

    results = {}

    # 1. supervised classifier
    clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=SEED
    )
    clf.fit(Xtr, y[train])
    val_scores = clf.predict_proba(Xva)[:, 1]
    thr, val_f1 = best_f1_threshold(y[val].astype(int), val_scores)
    test_scores = clf.predict_proba(Xte)[:, 1]
    lr_metrics = classification_metrics(y[test].astype(int), test_scores, threshold=thr)
    results["finmind_fraud_ulb"] = {
        "model": "LogisticRegression (class_weight=balanced)",
        "trained_on_rows": int(train.sum()),
        "test_metrics": lr_metrics,
        "validation_tuned_threshold": thr,
        "validation_f1_at_threshold": val_f1,
    }
    print(f"    finmind_fraud_ulb (LR) test: f1={lr_metrics['f1']} pr_auc={lr_metrics['pr_auc']} roc_auc={lr_metrics['roc_auc']}")

    # 2. gradient boosting comparison
    hgb = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.1, random_state=SEED, class_weight="balanced"
    )
    hgb.fit(Xtr, y[train])
    val_scores_h = hgb.predict_proba(Xva)[:, 1]
    thr_h, val_f1_h = best_f1_threshold(y[val].astype(int), val_scores_h)
    hgb_metrics = classification_metrics(
        y[test].astype(int), hgb.predict_proba(Xte)[:, 1], threshold=thr_h
    )
    results["finmind_fraud_ulb_hgb"] = {
        "model": "HistGradientBoostingClassifier",
        "trained_on_rows": int(train.sum()),
        "test_metrics": hgb_metrics,
        "validation_tuned_threshold": thr_h,
        "validation_f1_at_threshold": val_f1_h,
    }
    print(f"    finmind_fraud_ulb_hgb test: f1={hgb_metrics['f1']} pr_auc={hgb_metrics['pr_auc']} roc_auc={hgb_metrics['roc_auc']}")

    # 3. unsupervised IF fitted on TRAIN NORMALS only (old artifact's flaw fixed)
    normals_train = Xtr[y[train] == 0]
    iforest = IsolationForest(
        contamination=float(y[train].mean()), random_state=SEED, n_estimators=200
    )
    iforest.fit(normals_train)
    if_scores = -iforest.score_samples(Xte)
    if_pred = (iforest.predict(Xte) == -1).astype(int)
    if_metrics = classification_metrics(y[test].astype(int), if_scores, y_pred=if_pred)
    results["finmind_fraud_ulb_iforest"] = {
        "model": "IsolationForest (fit on train normals only)",
        "trained_on_rows": int(normals_train.shape[0]),
        "test_metrics": if_metrics,
    }
    print(f"    finmind_fraud_ulb_iforest test: f1={if_metrics['f1']} pr_auc={if_metrics['pr_auc']} roc_auc={if_metrics['roc_auc']}")

    # ---- save artifacts + registry ----
    artifacts = [
        ("finmind_fraud_ulb", clf, results["finmind_fraud_ulb"], {"scaler": scaler}, "validated"),
        ("finmind_fraud_ulb_hgb", hgb, results["finmind_fraud_ulb_hgb"], {"scaler": scaler}, "validated"),
        ("finmind_fraud_ulb_iforest", iforest, results["finmind_fraud_ulb_iforest"], {"scaler": scaler}, "evaluation_only"),
    ]
    for name, model_obj, payload, extra, status in artifacts:
        model_dir = f"data/models/{name}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model_obj, model_path)
        if "scaler" in extra:
            joblib.dump(extra["scaler"], os.path.join(model_dir, "scaler.pkl"))
        write_json(os.path.join(model_dir, "config.json"), {
            "model": payload["model"], "seed": SEED,
            "features": used_features, "target": "Class",
            "split": "chronological on Time (70/15/15); scaler fit on train only",
            "trained_rows": payload["trained_on_rows"],
            "trained_at": utc_now_iso(),
        })
        write_json(os.path.join(model_dir, "metrics.json"), payload)
        registry.register({
            "model_name": name,
            "model_type": payload["model"],
            "task": "fraud_detection",
            "dataset": "ULB Credit Card Fraud (mlg-ulb/creditcardfraud)",
            "version": "1.0",
            "trained_at": utc_now_iso(),
            "features": used_features,
            "target": "Class",
            "hyperparameters": {"random_state": SEED, "time_feature_excluded": True},
            "metrics": payload["test_metrics"],
            "artifact_path": model_path,
            "checksum": file_checksum(model_path),
            "status": status,
            "leakage_audit": dup_audit,
            "environment": env,
        })

    return {
        "ulb": {
            "split": {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum()),
                       "train_fraud": int(y[train].sum()), "val_fraud": int(y[val].sum()),
                       "test_fraud": int(y[test].sum())},
            "duplicate_cross_split_audit": dup_audit,
            "models": results,
        }
    }


def main() -> None:
    print("=" * 70)
    print("FINMIND FRAUD / ANOMALY MODEL TRAINING")
    print("=" * 70)
    env = environment_snapshot()
    registry = ModelRegistry()
    report = {
        "task": "fraud_anomaly",
        "trained_at": utc_now_iso(),
        "random_seed": SEED,
        "split_fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC,
                             "test": round(1 - TRAIN_FRAC - VAL_FRAC, 2)},
        "environment": env,
    }
    report.update(train_paysim(registry, env))
    report.update(train_ulb(registry, env))
    registry.save()
    write_json("data/reports/training/fraud_training_report.json", report)
    print("\nArtifacts: data/models/finmind_{anomaly,fraud_paysim,fraud_ulb,fraud_ulb_hgb,fraud_ulb_iforest}")
    print("Registry:  data/models/registry.json")


if __name__ == "__main__":
    main()
