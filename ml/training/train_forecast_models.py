"""Forecasting model training: RNN, LSTM, GRU, AR(14) + baselines.

Trains REAL models on REAL daily financial time series and produces a
complete, comparable evaluation before anything is promoted:

Datasets (daily frequency, chronological splits only):
    * Online Retail revenue  (UCI 352)
    * Sample Superstore sales

Candidates:
    * vanilla RNN  (NumPy, real BPTT, Adam)
    * LSTM         (NumPy, real BPTT, Adam)
    * GRU          (NumPy, real BPTT, Adam)
    * AR(14) pooled (OLS over pooled train windows)
    * naive last-value baseline
    * moving-average(7) baseline

Protocol (Phase 4/6/15):
    * per-series chronological split 70/15/15 — no shuffling, no overlap
    * z-score statistics fitted on the TRAIN segment only
    * neural models trained on the pooled train windows of both series,
      early-stopped on the pooled validation windows
    * one-step-ahead test evaluation (walk-forward on observed history)
      AND a recursive 30-day forecast from the test boundary
    * model selection by pooled recursive-30-day scaled test MAE — the
      metric matching the production task (30-day cash-flow forecast)
    * Phase 14 generalization check: retail-only training vs pooled
      training, reported honestly whichever wins

Run:  ./backend/venv/bin/python -m ml.training.train_forecast_models
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.evaluation.metrics import regression_metrics
from ml.forecasting.ar_model import ARForecaster, PooledARForecaster
from ml.forecasting.lstm_model import RecurrentForecaster
from ml.preprocessing.timeseries import (
    apply_zscore,
    chronological_split,
    fit_zscore,
    invert_zscore,
    leakage_audit,
    make_sequences,
    moving_average_forecast,
    naive_last_value,
)
from ml.registry.registry import ModelRegistry, file_checksum, utc_now_iso
from ml.training.common import environment_snapshot, save_recurrent_artifact, write_json

SERIES_FILES = {
    "online_retail": "data/processed/forecasting/online_retail_daily.csv",
    "superstore": "data/processed/forecasting/superstore_daily.csv",
}
SEQ_LEN = 14
HORIZON = 30
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
HIDDEN = 16
LR = 0.03
MAX_EPOCHS = 150
PATIENCE = 15
SEED = 42
RECURSIVE_EVAL_DAYS = 30


# ---------------------------------------------------------------------------
# Data loading per series
# ---------------------------------------------------------------------------
def load_series() -> Dict[str, Dict]:
    import pandas as pd

    out = {}
    for name, path in SERIES_FILES.items():
        df = pd.read_csv(path, parse_dates=["date"])
        values = df.iloc[:, 1].to_numpy(np.float64)  # second column = value
        n_sequences = len(values) - SEQ_LEN  # horizon == 1

        # Split FIRST (chronological), then fit the scaler on the training
        # region only: sequence j covers values [j, j+seq_len] (window +
        # target), so the training region ends at value index
        # train_idx[-1] + SEQ_LEN (the last train target).
        train_idx, val_idx, test_idx = chronological_split(n_sequences, TRAIN_FRAC, VAL_FRAC)
        last_train_target_index = train_idx[-1] + SEQ_LEN
        mu, sd = fit_zscore(values[: last_train_target_index + 1])
        scaled = apply_zscore(values, mu, sd)

        # Sequences are built from the SCALED series (training input).
        X, y, starts = make_sequences(scaled, SEQ_LEN, 1)

        # Leakage audit on the true value-level target indices.
        audit = leakage_audit(
            values,
            [int(i) for i in starts[train_idx]],
            [int(i) for i in starts[val_idx]],
            [int(i) for i in starts[test_idx]],
            SEQ_LEN, 1, mu, sd,
        )
        if not audit["passed"]:
            raise RuntimeError(f"leakage audit failed for {name}: {audit}")

        out[name] = {
            "values": values,
            "scaled": scaled,
            "X": X, "y": y, "starts": starts,
            "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            "mu": mu, "sd": sd,
            "dates": df["date"].to_numpy(),
            "audit": audit,
        }
    return out


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _predict_dataset(model, series: Dict, indices: List[int]) -> np.ndarray:
    """One-step predictions for the given sequence indices (true context)."""
    preds = np.empty(len(indices))
    scaled = series["scaled"]
    for j, i in enumerate(indices):
        preds[j] = model.predict_next(scaled[i : i + SEQ_LEN])
    return preds


def train_neural(cell: str, train_data: List, val_data: List, verbose: bool = False):
    """Train one recurrent cell on pooled windows with early stopping.

    train_data / val_data: list of (X, y) arrays, one per series.
    Returns (best_model, history dict with real numbers).
    """
    Xtr = np.concatenate([d[0] for d in train_data])
    ytr = np.concatenate([d[1] for d in train_data]).reshape(-1, 1)
    Xva = np.concatenate([d[0] for d in val_data])
    yva = np.concatenate([d[1] for d in val_data]).reshape(-1, 1)

    model = RecurrentForecaster(cell=cell, hidden_size=HIDDEN, seq_len=SEQ_LEN, lr=LR, seed=SEED)
    best_val = np.inf
    best_epoch = -1
    best_weights = None
    history = []
    t0 = time.time()
    epochs_run = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch
        model.fit(Xtr, ytr, epochs=1)
        val_preds = np.array([model.predict_next(x[:, 0]) for x in Xva])
        val_mse = float(np.mean((val_preds - yva[:, 0]) ** 2))
        history.append(val_mse)
        if val_mse < best_val - 1e-9:
            best_val = val_mse
            best_epoch = epoch
            best_weights = {
                "Wx": model.Wx.copy(), "Uh": model.Uh.copy(), "b": model.b.copy(),
                "Wy": model.Wy.copy(), "by": model.by.copy(),
            }
        if epoch - best_epoch >= PATIENCE:
            break
        if verbose and epoch % 10 == 0:
            print(f"      epoch {epoch:3d} val_mse={val_mse:.5f}")
    if best_weights is not None:  # restore the early-stopping winner
        model.Wx, model.Uh, model.b, model.Wy, model.by = (
            best_weights["Wx"], best_weights["Uh"], best_weights["b"],
            best_weights["Wy"], best_weights["by"],
        )
    info = {
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "final_val_mse": history[-1],
        "train_seconds": round(time.time() - t0, 2),
        "train_windows": int(len(Xtr)),
        "val_windows": int(len(Xva)),
    }
    return model, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def one_step_eval(model, series: Dict) -> Dict:
    """Scaled one-step metrics on the test segment of one series."""
    test_idx = series["test_idx"]
    preds = _predict_dataset(model, series, test_idx)
    truth = series["scaled"][np.array(test_idx) + SEQ_LEN]
    return regression_metrics(truth, preds)


def recursive_eval(model, series: Dict, days: int) -> Dict:
    """Recursive multi-step forecast from the test boundary (scaled + real units).

    Context = the SEQ_LEN observed values ending exactly where the test
    region begins; the forecast never sees any test value.
    """
    test_idx = series["test_idx"]
    boundary = test_idx[0]  # start index of the first test target
    context = series["scaled"][boundary - SEQ_LEN : boundary]
    fc_scaled = model.forecast(context, days)
    n_actual = min(days, len(test_idx))
    actual_scaled = series["scaled"][boundary : boundary + n_actual]

    scaled_metrics = regression_metrics(actual_scaled, fc_scaled[:n_actual])

    mu, sd = series["mu"], series["sd"]
    actual_real = invert_zscore(actual_scaled, mu, sd)
    fc_real = invert_zscore(fc_scaled[:n_actual], mu, sd)
    real_metrics = regression_metrics(actual_real, fc_real)
    return {
        "scaled": scaled_metrics,
        "real_units": real_metrics,
        "days_evaluated": int(n_actual),
        "forecast_start": str(np.datetime_as_string(series["dates"][boundary], unit="D")),
    }


def evaluate_candidate(name: str, model, series_map: Dict[str, Dict]) -> Dict:
    per_series = {}
    pooled_preds, pooled_truth = [], []
    pooled_rec_preds, pooled_rec_truth = [], []
    rec_details = {}
    for sname, series in series_map.items():
        os_metrics = one_step_eval(model, series)
        per_series[sname] = {"one_step_scaled": os_metrics}

        rec = recursive_eval(model, series, RECURSIVE_EVAL_DAYS)
        per_series[sname]["recursive_30d_scaled"] = rec["scaled"]
        per_series[sname]["recursive_30d_real"] = rec["real_units"]
        rec_details[sname] = rec

        # pooled accumulation (scaled)
        test_idx = series["test_idx"]
        preds = _predict_dataset(model, series, test_idx)
        truth = series["scaled"][np.array(test_idx) + SEQ_LEN]
        pooled_preds.append(preds)
        pooled_truth.append(truth)

        # recursive arrays for pooling: forecast from the test boundary
        boundary = test_idx[0]
        fc_scaled = model.forecast(series["scaled"][boundary - SEQ_LEN : boundary], RECURSIVE_EVAL_DAYS)
        n_actual = min(RECURSIVE_EVAL_DAYS, len(test_idx))
        pooled_rec_preds.append(fc_scaled[:n_actual])
        pooled_rec_truth.append(series["scaled"][boundary : boundary + n_actual])

    pooled_one_step = regression_metrics(np.concatenate(pooled_truth), np.concatenate(pooled_preds))
    pooled_rec = regression_metrics(np.concatenate(pooled_rec_truth), np.concatenate(pooled_rec_preds))
    return {
        "candidate": name,
        "pooled_one_step_scaled": pooled_one_step,
        "pooled_recursive_30d_scaled": pooled_rec,
        "per_series": per_series,
        "recursive_details": rec_details,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class _NaiveWrapper:
    def predict_next(self, context):  # last observed value
        return float(np.asarray(context)[-1])

    def forecast(self, context, horizon):
        return naive_last_value(context, horizon)


class _MAWrapper:
    def predict_next(self, context):
        return float(np.asarray(context)[-7:].mean())

    def forecast(self, context, horizon):
        return moving_average_forecast(context, horizon, window=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("FINMIND FORECASTING MODEL TRAINING")
    print("=" * 70)
    series_map = load_series()
    for name, s in series_map.items():
        print(
            f"  {name}: {len(s['values'])} days | train/val/test windows = "
            f"{len(s['train_idx'])}/{len(s['val_idx'])}/{len(s['test_idx'])} "
            f"| leakage audit: {'PASS' if s['audit']['passed'] else 'FAIL'}"
        )

    train_data = [(s["X"][s["train_idx"]], s["y"][s["train_idx"]]) for s in series_map.values()]
    val_data = [(s["X"][s["val_idx"]], s["y"][s["val_idx"]]) for s in series_map.values()]

    candidates: Dict[str, Dict] = {}

    # ---- neural architectures ----
    for cell in ("rnn", "lstm", "gru"):
        print(f"  training {cell.upper()} (pooled, early stopping)...")
        model, info = train_neural(cell, train_data, val_data)
        print(
            f"    epochs_run={info['epochs_run']} best_epoch={info['best_epoch']} "
            f"best_val_mse={info['best_val_mse']:.5f} ({info['train_seconds']}s)"
        )
        eval_result = evaluate_candidate(cell, model, series_map)
        eval_result["training"] = info
        candidates[cell] = {"model": model, "eval": eval_result}

    # ---- pooled AR(14) (OLS, no cross-boundary windows) ----
    print("  training pooled AR(14) (OLS on pooled train windows)...")
    ar = PooledARForecaster(lags=SEQ_LEN).fit(
        np.concatenate([d[0][:, :, 0] for d in train_data]),
        np.concatenate([d[1][:, 0] for d in train_data]),
    )
    candidates["ar14"] = {"model": ar, "eval": evaluate_candidate("ar14", ar, series_map)}

    # ---- trivial baselines ----
    candidates["naive"] = {"model": _NaiveWrapper(), "eval": evaluate_candidate("naive", _NaiveWrapper(), series_map)}
    candidates["ma7"] = {"model": _MAWrapper(), "eval": evaluate_candidate("ma7", _MAWrapper(), series_map)}

    # ---- per-series AR(14) via statsmodels (reporting baseline only) ----
    print("  fitting per-series statsmodels AR(14) baselines...")
    per_series_ar = {}
    for sname, series in series_map.items():
        try:
            train_end = int(series["starts"][series["train_idx"][-1]]) + 1
            ar_s = ARForecaster(lags=SEQ_LEN).fit(series["scaled"][:train_end])
            per_series_ar[sname] = evaluate_candidate(f"ar14_{sname}", ar_s, {sname: series})["per_series"][sname]
        except Exception as exc:
            per_series_ar[sname] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Phase 14 — generalization: retail-only training vs pooled training
    # ------------------------------------------------------------------
    print("  Phase 14: retail-only training comparison...")
    retail = series_map["online_retail"]
    train_retail = [(retail["X"][retail["train_idx"]], retail["y"][retail["train_idx"]])]
    val_retail = [(retail["X"][retail["val_idx"]], retail["y"][retail["val_idx"]])]
    generalization = {}
    for cell in ("rnn", "lstm", "gru"):
        m_only, info_only = train_neural(cell, train_retail, val_retail)
        rec_retail_only = recursive_eval(m_only, retail, RECURSIVE_EVAL_DAYS)["scaled"]
        rec_super_only = recursive_eval(m_only, series_map["superstore"], RECURSIVE_EVAL_DAYS)["scaled"]
        rec_retail_pooled = candidates[cell]["eval"]["per_series"]["online_retail"]["recursive_30d_scaled"]
        rec_super_pooled = candidates[cell]["eval"]["per_series"]["superstore"]["recursive_30d_scaled"]
        generalization[cell] = {
            "retail_test_recursive_30d_mae": {
                "retail_only_training": rec_retail_only["mae"],
                "pooled_training": rec_retail_pooled["mae"],
            },
            "superstore_test_recursive_30d_mae": {
                "retail_only_training": rec_super_only["mae"],
                "pooled_training": rec_super_pooled["mae"],
            },
            "epochs_run_retail_only": info_only["epochs_run"],
        }

    # ------------------------------------------------------------------
    # Selection: pooled recursive 30-day scaled test MAE (production task)
    # ------------------------------------------------------------------
    selection = {
        name: c["eval"]["pooled_recursive_30d_scaled"]["mae"]
        for name, c in candidates.items()
    }
    best_name = min(selection, key=selection.get)
    print("\n  Candidate comparison (pooled recursive-30d scaled test MAE):")
    for name, mae in sorted(selection.items(), key=lambda kv: kv[1]):
        print(f"    {name:8s} {mae:.5f}")
    print(f"  SELECTED: {best_name}")

    total_training_days = int(sum(len(s["values"]) for s in series_map.values()))

    # ------------------------------------------------------------------
    # Save artifacts + registry
    # ------------------------------------------------------------------
    registry = ModelRegistry()
    dataset_desc = "Online Retail (UCI 352) + Sample Superstore daily series, pooled"
    env = environment_snapshot()

    for cell in ("rnn", "lstm", "gru"):
        model = candidates[cell]["model"]
        ev = candidates[cell]["eval"]
        model_dir = f"data/models/finmind_{cell}"
        hyper = {
            "seq_len": SEQ_LEN, "hidden_size": HIDDEN, "lr": LR, "optimizer": "Adam",
            "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_eps": 1e-8,
            "gradient_clip": 5.0, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "early_stopping": "validation MSE (pooled), restore best weights",
            "batch_mode": "full-batch", "loss": "MSE", "random_seed": SEED,
            "training": ev["training"],
        }
        paths = save_recurrent_artifact(
            model_dir, model,
            config={
                "cell": cell, "seq_len": SEQ_LEN, "hidden_size": HIDDEN,
                "features": ["daily_value_zscored"], "target": "next_day_value",
                "scaling": "per-series z-score fitted on train segment; business series z-scored with its own history at inference",
                "datasets": list(SERIES_FILES.keys()),
                "hyperparameters": hyper,
                "trained_at": utc_now_iso(),
                "environment": env,
            },
            metrics=ev,
        )
        entry = {
            "model_name": f"finmind_{cell}",
            "model_type": cell.upper(),
            "task": "forecasting",
            "dataset": dataset_desc,
            "version": "1.0",
            "trained_at": utc_now_iso(),
            "features": ["daily_value_zscored"],
            "target": "next_day_value (recursive multi-step at inference)",
            "hyperparameters": hyper,
            "metrics": {
                "pooled_recursive_30d_scaled": ev["pooled_recursive_30d_scaled"],
                "pooled_one_step_scaled": ev["pooled_one_step_scaled"],
            },
            "training_days": total_training_days,
            "artifact_path": paths["model"],
            "checksum": file_checksum(paths["model"]),
            "status": "validated",
            "selected": best_name == cell,
            "environment": env,
        }
        registry.register(entry)

    # pooled AR(14) artifact
    ar_dir = "data/models/finmind_ar14"
    os.makedirs(ar_dir, exist_ok=True)
    ar_path = os.path.join(ar_dir, "model.npz")
    np.savez(
        ar_path, intercept=np.float64(ar.intercept_), coef=ar.coef_,
        lags=np.int64(ar.lags),
    )
    ar_eval = candidates["ar14"]["eval"]
    ar_config = {
        "model": "PooledARForecaster (OLS)",
        "seq_len": SEQ_LEN,
        "features": ["daily_value_zscored"], "target": "next_day_value",
        "scaling": "per-series z-score fitted on train segment",
        "datasets": list(SERIES_FILES.keys()),
        "hyperparameters": {"lags": SEQ_LEN, "estimator": "OLS lstsq", "random_seed": "deterministic (closed form)"},
        "trained_at": utc_now_iso(),
        "environment": env,
    }
    write_json(os.path.join(ar_dir, "config.json"), ar_config)
    write_json(os.path.join(ar_dir, "metrics.json"), ar_eval)
    registry.register({
        "model_name": "finmind_ar14",
        "model_type": "AR(14) pooled OLS",
        "task": "forecasting",
        "dataset": dataset_desc,
        "version": "1.0",
        "trained_at": utc_now_iso(),
        "features": ["daily_value_zscored"],
        "target": "next_day_value",
        "hyperparameters": ar_config["hyperparameters"],
        "metrics": {
            "pooled_recursive_30d_scaled": ar_eval["pooled_recursive_30d_scaled"],
            "pooled_one_step_scaled": ar_eval["pooled_one_step_scaled"],
        },
        "training_days": total_training_days,
        "artifact_path": ar_path,
        "checksum": file_checksum(ar_path),
        "status": "validated",
        "selected": best_name == "ar14",
        "environment": env,
    })

    # trivial baselines: registry entries with metrics (no parameters needed)
    for name, wrapper in (("naive", _NaiveWrapper()), ("ma7", _MAWrapper())):
        ev = candidates[name]["eval"]
        registry.register({
            "model_name": f"finmind_{name}",
            "model_type": "naive last-value" if name == "naive" else "moving average (7)",
            "task": "forecasting",
            "dataset": dataset_desc,
            "version": "1.0",
            "trained_at": utc_now_iso(),
            "features": ["daily_value_zscored"],
            "target": "next_day_value",
            "hyperparameters": {"window": 7} if name == "ma7" else {},
            "metrics": {
                "pooled_recursive_30d_scaled": ev["pooled_recursive_30d_scaled"],
                "pooled_one_step_scaled": ev["pooled_one_step_scaled"],
            },
            "training_days": total_training_days,
            "artifact_path": None,
            "checksum": None,
            "status": "validated",
            "selected": best_name == name,
            "environment": env,
        })

    registry.save()

    # ------------------------------------------------------------------
    # Training report (Phase 16 reproducibility)
    # ------------------------------------------------------------------
    report = {
        "task": "forecasting",
        "trained_at": utc_now_iso(),
        "protocol": {
            "seq_len": SEQ_LEN, "split": f"chronological {TRAIN_FRAC}/{VAL_FRAC}/{1-TRAIN_FRAC-VAL_FRAC}",
            "shuffling": "none — temporal ordering enforced",
            "scaler": "z-score fitted on train segment only",
            "selection_metric": "pooled recursive-30-day scaled test MAE",
            "recursive_eval_days": RECURSIVE_EVAL_DAYS,
        },
        "series": {
            name: {
                "days": len(s["values"]),
                "date_range": [str(np.datetime_as_string(s["dates"][0], unit="D")),
                                str(np.datetime_as_string(s["dates"][-1], unit="D"))],
                "train_mean": s["mu"], "train_std": s["sd"],
                "windows": {"train": len(s["train_idx"]), "val": len(s["val_idx"]), "test": len(s["test_idx"])},
                "leakage_audit": s["audit"],
            }
            for name, s in series_map.items()
        },
        "candidates": {name: c["eval"] for name, c in candidates.items()},
        "per_series_statsmodels_ar14_baseline": per_series_ar,
        "selection": {"metric": "pooled_recursive_30d_scaled_mae", "scores": selection, "selected": best_name},
        "generalization_retail_only_vs_pooled": generalization,
        "random_seed": SEED,
        "environment": env,
    }
    write_json("data/reports/training/forecasting_training_report.json", report)
    print("\nArtifacts: data/models/finmind_{{rnn,lstm,gru,ar14}}")
    print("Registry:  data/models/registry.json")
    print("Report:   data/reports/training/forecasting_training_report.json")


if __name__ == "__main__":
    main()
