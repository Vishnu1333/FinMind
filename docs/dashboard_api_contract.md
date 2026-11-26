# FinMind Dashboard — Backend API Contract

> Status of this document: the **backend prediction pipeline is healthy and
> fully artifact-backed** (see `data/models/registry.json`). The known
> Dashboard loading/data-display issue is a **frontend problem and is
> intentionally NOT addressed here** — no mock data has been added anywhere
> to hide it. This contract exists so the frontend fix can consume the real
> endpoints below.

All endpoints require the auth cookie/header used by the rest of the app
(`get_current_user`), and every value is business-scoped — a user only ever
sees their own business's data. All monetary amounts are in the business
currency (`MYR`), returned as JSON numbers or decimal strings.

## 1. Balance, income, expenses, net — `GET /analytics/summary`

Query params (optional): `start`, `end` (ISO datetimes; default period when omitted).

```json
{
  "currency": "MYR",
  "accounts": [
    {"account_id": 1, "name": "Main Operating", "balance": "42150.00"}
  ],
  "total_balance": "42150.00",
  "total_income": "18600.00",
  "total_expenses": "14250.00",
  "net_cash_flow": "4350.00"
}
```

## 2. Cash flow (period) — `GET /analytics/cash-flow`

Query params (optional): `start`, `end`.

```json
{
  "period_start": "2026-09-04T00:00:00",
  "period_end": "2026-10-04T23:59:59",
  "income": "18600.00",
  "expenses": "14250.00",
  "net": "4350.00"
}
```

## 3. Daily cash-flow series — `GET /analytics/cash-flow/daily`

Query params: `days` (int, default 90, range 7–365).

```json
{
  "currency": "MYR",
  "days": 90,
  "series": [
    {"date": "2026-09-04", "income": "950.00", "expenses": "300.00", "net": "650.00"}
  ]
}
```

## 4. Cash runway — `GET /analytics/runway`

```json
{"months": "6.4", "insufficient_data": false, "message": null}
```
With insufficient expense history: `{"months": null, "insufficient_data": true, "message": "..."}`

## 5. ML forecast — `GET /predictions/forecast`

Query params: `days` (int, default 30, capped at 90).

Served by the **registry-selected trained model artifact**
(`data/models/finmind_gru/` at the time of writing — selected by offline
walk-forward evaluation, not hardcoded). The model is **loaded from disk
and verified by checksum; no training happens during a request**.

```json
{
  "currency": "MYR",
  "forecast": [
    {"day": 1, "date": "2026-10-05", "predicted_cashflow": 412.55, "cumulative": 412.55},
    {"day": 2, "date": "2026-10-06", "predicted_cashflow": 388.10, "cumulative": 800.65}
  ],
  "model": {
    "selected": "GRU",
    "selection_method": "offline walk-forward evaluation on held-out daily financial series; selected by pooled recursive-30-day scaled test MAE (see data/models/registry.json)",
    "artifact": "finmind_gru",
    "version": "1.0",
    "artifact_checksum_verified": true,
    "trained_on_days": 1832,
    "training_datasets": "Online Retail (UCI 352) + Sample Superstore daily series, pooled",
    "business_context_days": 180,
    "context_window": 14,
    "holdout_days": 60,
    "holdout_mae_scaled": 0.7512,
    "holdout_rmse_scaled": 0.9276,
    "candidates": {
      "finmind_rnn": {"holdout_mae_scaled": 0.8053, "holdout_rmse_scaled": 1.006},
      "finmind_lstm": {"holdout_mae_scaled": 0.9078, "holdout_rmse_scaled": 1.0474},
      "finmind_gru":  {"holdout_mae_scaled": 0.7512, "holdout_rmse_scaled": 0.9276},
      "finmind_ar14": {"holdout_mae_scaled": 0.8746, "holdout_rmse_scaled": 1.0567},
      "finmind_ma7":  {"holdout_mae_scaled": 0.8581, "holdout_rmse_scaled": 1.0237},
      "finmind_naive":{"holdout_mae_scaled": 1.3612, "holdout_rmse_scaled": 1.8338}
    },
    "series_mean": 415.2,
    "series_std": 388.7
  }
}
```

Error semantics (honest, never fabricated):
- `422` — business has fewer than 60 days of history (message explains what is needed)
- `500` — artifact/registry problems (message names the missing artifact)

## 6. Anomalies — `GET /predictions/anomalies`

Query params: `top_n` (default 25), `min_amount` (optional).

Served by the **globally-trained IsolationForest artifact**
(`finmind_anomaly`, trained on 6,082,007 PaySim transactions with the same
feature schema used here) plus per-business statistical evidence. No model
is fitted during a request.

```json
{
  "available": true,
  "reason": null,
  "anomalies": [
    {
      "transaction_id": 482,
      "date": "2026-08-26T03:12:00",
      "type": "DEBIT",
      "amount": 8950.0,
      "merchant": "TechSupply Sdn Bhd",
      "category": "Equipment",
      "anomaly_score": -0.5123,
      "reasons": ["about 22× your typical payment"],
      "evidence": {
        "business_avg_spend": 405.11,
        "business_spend_std": 290.4,
        "amount_z_score": 2.31,
        "hour": 3,
        "duplicate": false
      }
    }
  ],
  "model": {
    "type": "IsolationForest",
    "artifact": "finmind_anomaly",
    "trained_on": 6082007,
    "training_dataset": "PaySim synthetic mobile money transactions (ealaxi/paysim1)",
    "decision_rule": "flag = artifact score below the 5th percentile of this business's own transaction scores ...",
    "features": ["log_amount", "hour", "day_of_week", "is_weekend"],
    "evidence_rules": ["..."]
  }
}
```

If the business has < 40 transactions: `{"available": false, "reason": "...", "anomalies": [], "model": null}`.

## 7. Single-transaction anomaly check — `POST /predictions/anomaly`

Body: `{"amount": 8950.0}` → `{"available": true, "is_anomaly": true, "score": -0.5123, ...}`

## Model / dataset provenance

- Model registry (checksums, metrics, training runs): `data/models/registry.json`
- Dataset manifests + licenses: `data/manifests/`
- Data quality + preprocessing reports: `data/reports/quality/`
- Training + generalization reports: `data/reports/training/`, `data/reports/generalization_report.json`
