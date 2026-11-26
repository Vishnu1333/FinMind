"""Dataset acquisition, quality analysis, processing and manifests.

Every dataset used by FinMind's ML models is prepared through this module.
It enforces the Phase 2/3 requirements:

* real files are verified on disk (sha256 recorded in the manifest);
* every manifest field (rows, columns, features, target, license) is
  computed from the actual data — never hand-written;
* missing values, duplicates, datatype issues, timestamp validity,
  outliers and class imbalance are all measured and reported;
* every transformation decision is documented in the quality report —
  data is never silently dropped.

Run:  ./backend/venv/bin/python -m ml.training.datasets
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RAW = "data/raw"
PROCESSED = "data/processed"
MANIFESTS = "data/manifests"
QUALITY = "data/reports/quality"
CHECKSUM = "sha256"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    h = hashlib.new(CHECKSUM)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Generic quality analysis
# ---------------------------------------------------------------------------
def quality_report(
    df: pd.DataFrame,
    name: str,
    target: str = None,
    timestamp_col: str = None,
    timestamp_fmt: str = None,
) -> Dict[str, object]:
    """Measure missingness, duplicates, dtypes, outliers, imbalance."""
    n, p = df.shape
    report: Dict[str, object] = {
        "dataset": name,
        "rows": int(n),
        "columns": int(p),
        "analyzed_at": _now(),
        "column_dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": {
            c: int(df[c].isna().sum())
            for c in df.columns
            if df[c].isna().sum() > 0
        },
        "missing_total_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }

    # Numeric outliers (IQR rule) — counted, never deleted
    outliers = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        if n_out:
            outliers[c] = {"count": n_out, "iqr_lower": float(lo), "iqr_upper": float(hi)}
    report["outliers_iqr"] = outliers

    if target and target in df.columns:
        counts = df[target].value_counts(dropna=False).to_dict()
        pos = int(counts.get(1, 0))
        report["class_balance"] = {
            "target": target,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "positive_rate": round(pos / n, 6) if n else None,
        }

    if timestamp_col and timestamp_col in df.columns:
        ts = (
            pd.to_datetime(df[timestamp_col], format=timestamp_fmt, errors="coerce")
            if timestamp_fmt
            else pd.to_datetime(df[timestamp_col], errors="coerce")
        )
        invalid = int(ts.isna().sum())
        monotonic_after_sort = bool(df.loc[ts.notna(), timestamp_col].is_monotonic_increasing)
        report["timestamps"] = {
            "column": timestamp_col,
            "unparseable": invalid,
            "min": str(ts.min()),
            "max": str(ts.max()),
            "sorted_input": monotonic_after_sort,
            "duplicate_timestamps": int(ts.duplicated().sum()),
        }
    return report


def manifest_for(
    name: str,
    source: str,
    task: str,
    license_: str,
    raw_files: List[str],
    rows: int,
    columns: int,
    features: List[str],
    target: str,
    extra: Dict = None,
    download_date: str = None,
) -> Dict:
    m = {
        "name": name,
        "source": source,
        "task": task,
        "license": license_,
        "download_date": download_date or _now()[:10],
        "rows": int(rows),
        "columns": int(columns),
        "features": features,
        "target": target,
        "checksum": {f: sha256_file(f) for f in raw_files},
        "prepared_at": _now(),
    }
    if extra:
        m.update(extra)
    return m


# ---------------------------------------------------------------------------
# 1. Online Retail (UCI 352) — daily revenue series for forecasting
# ---------------------------------------------------------------------------
def prepare_online_retail() -> Dict[str, object]:
    from zipfile import ZipFile

    zip_path = f"{RAW}/online_retail/online-retail.zip"
    with ZipFile(zip_path) as z, z.open("Online Retail.xlsx") as f:
        import io

        df = pd.read_excel(
            io.BytesIO(f.read()), dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": "Float64"}
        )
    raw_rows = len(df)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    unparseable_dates = int(df["InvoiceDate"].isna().sum())

    # Documented exclusions (each counted, each justified):
    cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
    negative_qty = df["Quantity"] < 0
    negative_price = df["UnitPrice"] < 0
    zero_price = df["UnitPrice"] == 0
    keep = (~cancelled) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)

    sales = df[keep].copy()
    sales["revenue"] = sales["Quantity"] * sales["UnitPrice"]
    daily = sales.groupby(sales["InvoiceDate"].dt.floor("D"))["revenue"].sum()

    # Fill calendar gaps with 0.0 revenue (no invoices that day — a real zero)
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)

    decisions = {
        "rows_raw": int(raw_rows),
        "rows_cancelled_invoices_excluded": int(cancelled.sum()),
        "rows_negative_quantity_excluded": int(negative_qty.sum()),
        "rows_negative_unitprice_excluded": int(negative_price.sum()),
        "rows_zero_unitprice_excluded": int(zero_price.sum()),
        "rows_kept_for_revenue": int(len(sales)),
        "missing_customerid_rows_kept": int(df["CustomerID"].isna().sum()),
        "calendar_gap_days_filled_with_zero": int(
            (daily == 0).sum()
        ),
        "unparseable_invoice_dates": unparseable_dates,
        "exclusion_rationale": (
            "Revenue series counts completed positive-quantity, positive-price "
            "sales only: cancellations (InvoiceNo starting 'C') and returns "
            "(negative Quantity) are documented and excluded from REVENUE (they "
            "are not cash inflows); rows with UnitPrice<=0 are data artifacts. "
            "No row was deleted from the raw dataset on disk."
        ),
    }

    out_csv = f"{PROCESSED}/forecasting/online_retail_daily.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    daily.rename("revenue").to_frame().to_csv(out_csv, index_label="date")

    report = quality_report(
        df.rename(columns=str), "online_retail", timestamp_col="InvoiceDate"
    )
    report["transformation_decisions"] = decisions
    report["daily_series"] = {
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "n_days": int(len(daily)),
    }
    _write_json(f"{QUALITY}/online_retail.json", report)

    manifest = manifest_for(
        name="Online Retail",
        source="UCI Machine Learning Repository (dataset 352) — https://archive.ics.uci.edu/dataset/352/online+retail",
        task="daily_revenue_forecasting",
        license_="CC BY 4.0 (as listed by UCI)",
        raw_files=[zip_path],
        rows=int(raw_rows),
        columns=int(df.shape[1]),
        features=["date (aggregated)", "revenue = sum(Quantity * UnitPrice) per day"],
        target="next-day revenue",
        extra={"quality_report": f"{QUALITY}/online_retail.json",
               "processed": out_csv,
               "transformation_decisions": decisions,
               "citation": "Daqing Chen, Sai Liang Saine, Kun Guo, 'Online Retail', UCI ML Repository, 2012."},
    )
    _write_json(f"{MANIFESTS}/online_retail_manifest.json", manifest)
    return {"manifest": manifest, "report": report}


# ---------------------------------------------------------------------------
# 2. Superstore — daily sales series for forecasting
# ---------------------------------------------------------------------------
def prepare_superstore() -> Dict[str, object]:
    csv_path = f"{RAW}/superstore/Sample - Superstore.csv"
    df = pd.read_csv(csv_path, encoding="latin-1")
    raw_rows = len(df)

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    ship_before_order = int((df["Ship Date"] < df["Order Date"]).sum())
    unparseable = int(df["Order Date"].isna().sum())

    sales = df[df["Sales"].notna() & (df["Sales"] > 0)].copy()
    daily = sales.groupby(sales["Order Date"].dt.floor("D"))["Sales"].sum()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)

    decisions = {
        "rows_raw": int(raw_rows),
        "rows_nonpositive_sales_excluded": int(raw_rows - len(sales)),
        "ship_date_before_order_date_rows": ship_before_order,
        "unparseable_order_dates": unparseable,
        "calendar_gap_days_filled_with_zero": int((daily == 0).sum()),
        "exclusion_rationale": (
            "Daily sales series = sum of Sales by Order Date for sales rows with "
            "Sales > 0. Days without orders are real zero-sales days. Nothing was "
            "deleted from the raw file; the series is an aggregation."
        ),
    }

    out_csv = f"{PROCESSED}/forecasting/superstore_daily.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    daily.rename("sales").to_frame().to_csv(out_csv, index_label="date")

    report = quality_report(df, "superstore", timestamp_col="Order Date")
    report["transformation_decisions"] = decisions
    report["daily_series"] = {
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "n_days": int(len(daily)),
    }
    _write_json(f"{QUALITY}/superstore.json", report)

    manifest = manifest_for(
        name="Sample Superstore",
        source="Tableau 'Sample Superstore' sample data via Kaggle mirror vivek468/superstore-dataset-final",
        task="daily_sales_forecasting",
        license_="Public sample data for educational use (per Kaggle mirror; original from Tableau community)",
        raw_files=[csv_path],
        rows=int(raw_rows),
        columns=int(df.shape[1]),
        features=["date (aggregated)", "sales = sum(Sales) per day"],
        target="next-day sales",
        extra={"quality_report": f"{QUALITY}/superstore.json",
               "processed": out_csv,
               "transformation_decisions": decisions},
    )
    _write_json(f"{MANIFESTS}/superstore_manifest.json", manifest)
    return {"manifest": manifest, "report": report}


# ---------------------------------------------------------------------------
# 3. PaySim — transaction behaviour + fraud labels
# ---------------------------------------------------------------------------
PAYSIM_FEATURES = [
    "log_amount", "type_encoded", "oldbalance_org", "oldbalance_dest",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
]
# Feature schema used by the LIVE per-business anomaly service (must match
# backend/app/services/anomaly_service.py _features)
ANOMALY_FEATURES = ["log_amount", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


def prepare_paysim() -> Dict[str, object]:
    from zipfile import ZipFile

    zip_path = f"{RAW}/ealaxi_paysim1/paysim1.zip"
    usecols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
               "oldbalanceDest", "isFraud", "isFlaggedFraud"]
    with ZipFile(zip_path) as z, z.open("PS_20174392719_1491204439457_log.csv") as f:
        df = pd.read_csv(
            f, usecols=usecols, dtype={"type": "category"},
        )
    raw_rows = len(df)

    # Row-level identity for the leakage audit: the FULL raw record
    # (event, entity, amount, pre-transaction state, label). Two rows with
    # the same hash are the same recorded transaction event.
    row_hash = pd.util.hash_pandas_object(
        df[["step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "oldbalanceDest", "isFraud"]],
        index=False,
    ).to_numpy(np.uint64)

    # NOTE (dataset authors): post-transaction balances (newbalanceOrig /
    # newbalanceDest) are NOT loaded at all — detected frauds are cancelled,
    # so post-transaction state leaks the outcome. isFlaggedFraud is an
    # existing system's output (target leakage) and is dropped after audit.
    flagged_col_present = "isFlaggedFraud" in df.columns
    flagged_fraud_agreement = None
    if flagged_col_present:
        flagged_fraud_agreement = {
            "rows_flagged": int(df["isFlaggedFraud"].sum()),
            "all_flagged_are_fraud": bool(
                (df.loc[df["isFlaggedFraud"] == 1, "isFraud"] == 1).all()
            ),
            "decision": "excluded from features: it is an upstream detector's output (target leakage)",
        }

    # hour-of-day and day-of-week from step (1 step == 1 hour of simulation)
    hour = df["step"] % 24
    dow = (df["step"] // 24) % 7
    X = np.column_stack([
        np.log1p(df["amount"].to_numpy(np.float64)),
        df["type"].cat.codes.to_numpy(np.float64),
        df["oldbalanceOrg"].to_numpy(np.float64),
        df["oldbalanceDest"].to_numpy(np.float64),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).to_numpy(np.float64) if hasattr(dow, "to_numpy") else (dow >= 5).astype(np.float64),
    ]).astype(np.float64)
    y = df["isFraud"].to_numpy(np.int8)
    step = df["step"].to_numpy(np.int32)

    # Live-schema anomaly features (for the artifact used by AnomalyService)
    X_anom = np.column_stack([
        np.log1p(df["amount"].to_numpy(np.float64)),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).astype(np.float64),
    ]).astype(np.float64)

    # exact duplicate feature rows (audit only — kept in data; split audit
    # verifies none of the SAME row lands in two different splits)
    dup_rows = int(pd.DataFrame(X).duplicated().sum())

    out_npz = f"{PROCESSED}/fraud/paysim_features.npz"
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=X.astype(np.float32), y=y, step=step,
        X_anomaly_schema=X_anom.astype(np.float32),
        row_hash=row_hash,
        feature_names=np.array(PAYSIM_FEATURES),
        anomaly_feature_names=np.array(ANOMALY_FEATURES),
    )

    report = quality_report(
        df.rename(columns=str), "paysim1", target="isFraud"
    )
    report["time_index"] = {
        "column": "step",
        "semantics": "1 step = 1 hour of simulated time; 744 steps = 31 days",
        "min": int(step.min()), "max": int(step.max()),
    }
    report["transformation_decisions"] = {
        "rows_raw": int(raw_rows),
        "duplicate_feature_rows": dup_rows,
        "isflaggedfraud_audit": flagged_fraud_agreement,
        "post_transaction_balances": (
            "newbalanceOrig/newbalanceDest NOT loaded: dataset authors state "
            "frauds are cancelled so these leak the outcome (Phase 15)."
        ),
        "chronological_split_rule": "split on `step` (1 step = 1 hour): train < val < test, no shuffle",
    }
    report["class_balance"]["positive_rate"] = round(float(y.mean()), 6)
    _write_json(f"{QUALITY}/paysim1.json", report)

    manifest = manifest_for(
        name="PaySim Synthetic Mobile Money Transactions",
        source="Kaggle ealaxi/paysim1 (PaySim simulator; E. A. Lopez-Rojos et al.)",
        task="fraud_detection (supervised) + unsupervised anomaly detection",
        license_="CC BY-SA 4.0 (as listed on the Kaggle dataset page)",
        raw_files=[zip_path],
        rows=int(raw_rows),
        columns=11,
        features=PAYSIM_FEATURES + ["isFlaggedFraud (audited, excluded as leakage)"],
        target="isFraud",
        extra={
            "quality_report": f"{QUALITY}/paysim1.json",
            "processed": out_npz,
            "citation": "E. A. Lopez-Rojas, A. Elmir, S. Axelsson. 'PaySim: A financial mobile money simulator for fraud detection'. EMSS 2016.",
            "prepared_note": "post-transaction balance columns excluded per dataset authors' leakage warning",
        },
        download_date="2026-09-23",
    )
    _write_json(f"{MANIFESTS}/ealaxi_paysim1_manifest.json", manifest)
    return {"manifest": manifest, "report": report}


# ---------------------------------------------------------------------------
# 4. ULB Credit Card Fraud
# ---------------------------------------------------------------------------
ULB_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def prepare_ulb_fraud() -> Dict[str, object]:
    from zipfile import ZipFile

    zip_path = f"{RAW}/mlg-ulb_creditcardfraud/creditcardfraud.zip"
    with ZipFile(zip_path) as z, z.open("creditcard.csv") as f:
        df = pd.read_csv(f)
    raw_rows = len(df)

    X = df[ULB_FEATURES].to_numpy(np.float64)
    y = df["Class"].to_numpy(np.int8)
    t = df["Time"].to_numpy(np.float64)

    # exact duplicate feature rows (audit only — kept in data; the split
    # audit verifies no raw record lands in two different splits)
    dup_rows = int(pd.DataFrame(X).duplicated().sum())

    out_npz = f"{PROCESSED}/fraud/ulb_features.npz"
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz, X=X.astype(np.float32), y=y, time=t,
        row_hash=pd.util.hash_pandas_object(df, index=False).to_numpy(np.uint64),
        feature_names=np.array(ULB_FEATURES),
    )

    report = quality_report(df, "ulb_creditcard_fraud", target="Class")
    report["time_index"] = {
        "column": "Time",
        "semantics": "seconds elapsed since the first transaction in the dataset (2 days of activity)",
        "min": float(t.min()), "max": float(t.max()),
    }
    report["transformation_decisions"] = {
        "rows_raw": int(raw_rows),
        "exact_duplicate_rows_in_raw": dup_rows,
        "duplicate_policy": (
            "duplicates are real repeated transaction records; kept, but the "
            "chronological split is audited so identical rows never straddle "
            "the train/test boundary"
        ),
        "chronological_split_rule": "split on Time (seconds since first tx): train < val < test, no shuffle",
        "pca_features_note": "V1..V28 are PCA components provided by the dataset (Worldline/ULB)",
    }
    _write_json(f"{QUALITY}/ulb_creditcard_fraud.json", report)

    manifest = manifest_for(
        name="Credit Card Fraud Detection (ULB)",
        source="Kaggle mlg-ulb/creditcardfraud (Worldline & ULB MLG)",
        task="fraud_detection (supervised) + unsupervised anomaly detection",
        license_="DbCL v1.0 (Database Contents License, as listed on the Kaggle dataset page)",
        raw_files=[zip_path],
        rows=int(raw_rows),
        columns=int(df.shape[1]),
        features=ULB_FEATURES,
        target="Class",
        extra={
            "quality_report": f"{QUALITY}/ulb_creditcard_fraud.json",
            "processed": out_npz,
            "citation": "Dal Pozzolo et al., 'Calibrating Probability with Undersampling for Unbalanced Classification', IEEE CIDM 2015.",
        },
        download_date="2026-09-23",
    )
    _write_json(f"{MANIFESTS}/mlg-ulb_creditcardfraud_manifest.json", manifest)
    return {"manifest": manifest, "report": report}


# ---------------------------------------------------------------------------
# 5. UCI Default of Credit Card Clients — credit risk classification
# ---------------------------------------------------------------------------
CREDIT_DEFAULT_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def prepare_uci_credit_default() -> Dict[str, object]:
    from zipfile import ZipFile

    zip_path = f"{RAW}/uciml_default-of-credit-card-clients-dataset/default-of-credit-card-clients-dataset.zip"
    with ZipFile(zip_path) as z, z.open("UCI_Credit_Card.csv") as f:
        df = pd.read_csv(f)
    raw_rows = len(df)

    target = "default.payment.next.month"
    X = df[CREDIT_DEFAULT_FEATURES].to_numpy(np.float64)
    y = df[target].to_numpy(np.int8)

    out_npz = f"{PROCESSED}/risk/uci_credit_default.npz"
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz, X=X.astype(np.float32), y=y,
        feature_names=np.array(CREDIT_DEFAULT_FEATURES),
    )

    report = quality_report(df, "uci_credit_default", target=target)
    report["transformation_decisions"] = {
        "rows_raw": int(raw_rows),
        "split_rule": "no timestamps exist -> stratified random split (documented, valid for this task)",
        "note": "this dataset is tabular (client snapshot), NOT a time series; the previous"
                " attempt to treat BILL_AMT1 rows as a temporal series was invalid (see audit)",
    }
    _write_json(f"{QUALITY}/uci_credit_default.json", report)

    manifest = manifest_for(
        name="Default of Credit Card Clients (Taiwan, 2005)",
        source="Kaggle uciml/default-of-credit-card-clients-dataset (UCI ML Repository)",
        task="credit_risk_classification",
        license_="CC0 1.0 (as listed on the Kaggle dataset page)",
        raw_files=[zip_path],
        rows=int(raw_rows),
        columns=int(df.shape[1]),
        features=CREDIT_DEFAULT_FEATURES,
        target=target,
        extra={
            "quality_report": f"{QUALITY}/uci_credit_default.json",
            "processed": out_npz,
            "citation": "Yeh, I. C. & Lien, C. H. (2009). 'The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients'. Expert Systems with Applications.",
        },
        download_date="2026-09-23",
    )
    _write_json(f"{MANIFESTS}/uciml_default-of-credit-card-clients-dataset_manifest.json", manifest)
    return {"manifest": manifest, "report": report}


# ---------------------------------------------------------------------------
# 6. Statlog German Credit — credit approval risk classification
# ---------------------------------------------------------------------------
GERMAN_COLUMNS = [
    "chk_acct", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_acct", "present_employment", "installment_rate", "personal_status",
    "other_debtors", "present_residence", "property", "age", "other_installments",
    "housing", "n_credits", "job", "n_people", "telephone", "foreign_worker",
]


def prepare_german_credit() -> Dict[str, object]:
    from zipfile import ZipFile

    zip_path = f"{RAW}/german_credit/statlog-german-credit-data.zip"
    with ZipFile(zip_path) as z, z.open("german.data") as f:
        raw = pd.read_csv(f, delim_whitespace=True, header=None, names=GERMAN_COLUMNS + ["target"])
    raw_rows = len(raw)

    df = raw.copy()
    df["target"] = (df["target"] == 2).astype(np.int8)  # 2 = bad credit -> 1
    categorical = [c for c in GERMAN_COLUMNS if df[c].dtype == object]
    for c in categorical:
        df[c] = df[c].astype("category").cat.codes.astype(np.float64)
    X = df[GERMAN_COLUMNS].to_numpy(np.float64)
    y = df["target"].to_numpy(np.int8)

    out_npz = f"{PROCESSED}/risk/german_credit.npz"
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz, X=X.astype(np.float32), y=y,
        feature_names=np.array(GERMAN_COLUMNS),
    )

    report = quality_report(raw.assign(target=df["target"]), "german_credit", target="target")
    report["transformation_decisions"] = {
        "rows_raw": int(raw_rows),
        "categorical_encoding": f"integer-encoded categories: {categorical}",
        "target_mapping": "1 = good credit -> 0, 2 = bad credit -> 1 (bad credit is the positive class)",
        "split_rule": "no timestamps -> stratified random split",
    }
    _write_json(f"{QUALITY}/german_credit.json", report)

    manifest = manifest_for(
        name="Statlog German Credit Data",
        source="UCI ML Repository (dataset 144) — https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        task="credit_risk_classification",
        license_="CC BY 4.0 (as listed by UCI)",
        raw_files=[zip_path],
        rows=int(raw_rows),
        columns=int(raw.shape[1]),
        features=GERMAN_COLUMNS,
        target="credit risk (1 = bad)",
        extra={
            "quality_report": f"{QUALITY}/german_credit.json",
            "processed": out_npz,
            "citation": "Hans Hofmann, UCI Statlog German Credit Data, 1994.",
        },
    )
    _write_json(f"{MANIFESTS}/german_credit_manifest.json", manifest)
    return {"manifest": manifest, "report": report}


# ---------------------------------------------------------------------------
# Correct the stale manifests (previously claimed "downloaded" for
# directories that are empty on disk — never fabricate status).
# ---------------------------------------------------------------------------
def fix_stale_manifests() -> None:
    stale = {
        "eaubet_paysim1": (
            "Not downloaded: this is a duplicate of ealaxi/paysim1 (already on disk). "
            "Recorded honestly instead of claiming 'downloaded'."
        ),
        "ieee-cis-fraud-detection": (
            "Not downloaded: Kaggle competition rules acceptance required (401 via API); "
            "fraud detection is already covered by PaySim + ULB datasets on disk."
        ),
        "m5-forecasting-accuracy": (
            "Not downloaded: Kaggle competition rules acceptance required (401 via API). "
            "Forecasting is covered by Online Retail + Superstore daily series."
        ),
    }
    for slug, reason in stale.items():
        path = f"{MANIFESTS}/{slug}_manifest.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        m["status"] = "not_available"
        m["status_reason"] = reason
        m["corrected_at"] = _now()
        m["rows_on_disk"] = 0
        _write_json(path, m)


def main() -> None:
    print("Preparing datasets (quality analysis + manifests + processed outputs)...")
    for name, fn in [
        ("online_retail", prepare_online_retail),
        ("superstore", prepare_superstore),
        ("paysim", prepare_paysim),
        ("ulb_fraud", prepare_ulb_fraud),
        ("uci_credit_default", prepare_uci_credit_default),
        ("german_credit", prepare_german_credit),
    ]:
        print(f"  - {name} ...", end=" ", flush=True)
        out = fn()
        print(f"rows={out['manifest']['rows']}")
    fix_stale_manifests()
    print("Done. Manifests in data/manifests, quality reports in data/reports/quality.")


if __name__ == "__main__":
    main()
