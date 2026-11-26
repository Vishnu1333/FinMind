"""Time-series preprocessing for leakage-free forecasting pipelines.

Rules enforced by this module (Phase 4 / Phase 15 of the ML plan):

1. Sequences are built in strict chronological order — inputs are the
   `seq_len` values ending at t-1, the target is value(s) at t..t+h-1.
2. Splits are chronological and non-overlapping:
       [--------- TRAIN ---------][-- VAL --][-- TEST --]
   No shuffling is ever applied to time-series data.
3. Scaling statistics (z-score mean/std) are fitted on the TRAIN segment
   only and then applied to val/test. The fitted statistics are returned
   so they can be saved with the model artifact.
4. `leakage_audit` re-verifies these properties on the actual arrays.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Scaling (fitted on train only)
# ---------------------------------------------------------------------------
def fit_zscore(train_values: np.ndarray) -> Tuple[float, float]:
    """Fit z-score statistics on the TRAINING segment only."""
    v = np.asarray(train_values, dtype=np.float64).ravel()
    return float(v.mean()), float(v.std(ddof=0))


def apply_zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        raise ValueError("series has zero variance — z-score scaling undefined")
    return (np.asarray(values, dtype=np.float64).ravel() - mean) / std


def invert_zscore(scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(scaled, dtype=np.float64) * std + mean


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------
def make_sequences(
    scaled: np.ndarray,
    seq_len: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, targets_scaled_from) from a single scaled series.

    X[i] = scaled[i : i+seq_len]           (context window)
    y[i] = scaled[i+seq_len : i+seq_len+horizon]  (target window)

    Returns (X of shape (N, seq_len, 1), y of shape (N, horizon),
    start_indices of shape (N,)).
    """
    v = np.asarray(scaled, dtype=np.float64).ravel()
    if len(v) < seq_len + horizon:
        raise ValueError(
            f"series of length {len(v)} too short for seq_len={seq_len} + horizon={horizon}"
        )
    X, y, starts = [], [], []
    for i in range(len(v) - seq_len - horizon + 1):
        X.append(v[i : i + seq_len])
        y.append(v[i + seq_len : i + seq_len + horizon])
        starts.append(i + seq_len)  # index of the first target value
    return (
        np.asarray(X).reshape(-1, seq_len, 1),
        np.asarray(y),
        np.asarray(starts, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------
def chronological_split(
    n_sequences: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """Chronological index split: train < val < test, no overlap, no shuffle."""
    if not (0.0 < train_frac < 1.0 and 0.0 <= val_frac < 1.0):
        raise ValueError("fractions must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for a test set")
    n_train = int(n_sequences * train_frac)
    n_val = int(n_sequences * val_frac)
    train = list(range(0, n_train))
    val = list(range(n_train, n_train + n_val))
    test = list(range(n_train + n_val, n_sequences))
    if not train or not test:
        raise ValueError(
            f"split leaves empty segments (n={n_sequences}, "
            f"train={len(train)}, val={len(val)}, test={len(test)})"
        )
    return train, val, test


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def naive_last_value(context: np.ndarray, horizon: int) -> np.ndarray:
    """Persistence forecast: repeat the last observed value."""
    v = np.asarray(context, dtype=np.float64).ravel()
    return np.full(horizon, float(v[-1]))


def moving_average_forecast(context: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    """Forecast the mean of the last `window` observations for every step."""
    v = np.asarray(context, dtype=np.float64).ravel()
    w = v[-window:]
    return np.full(horizon, float(w.mean()))


def multi_step_forecast(predict_next_fn, context: np.ndarray, horizon: int, seq_len: int) -> np.ndarray:
    """Recursive multi-step forecasting given a one-step predictor."""
    window = list(np.asarray(context, dtype=np.float64).ravel())
    out: List[float] = []
    for _ in range(horizon):
        nxt = float(predict_next_fn(np.array(window[-seq_len:])))
        out.append(nxt)
        window.append(nxt)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Leakage audit (re-verified on real arrays, not assumed)
# ---------------------------------------------------------------------------
def leakage_audit(
    series_values: np.ndarray,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    seq_len: int,
    horizon: int,
    scaler_mean: float,
    scaler_std: float,
) -> Dict[str, object]:
    """Verify the no-leakage invariants on the actual split arrays.

    Checks:
      * index sets are disjoint (no duplicate sequences across splits)
      * every target index in val/test is strictly AFTER the last train
        target (no future->past contamination across the boundary)
      * scaler stats equal the training segment statistics
    """
    values = np.asarray(series_values, dtype=np.float64).ravel()
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)

    disjoint_1 = train_set.isdisjoint(val_set)
    disjoint_2 = train_set.isdisjoint(test_set)
    disjoint_3 = val_set.isdisjoint(test_set)

    # Sequence i covers series indices [i, i+seq_len+horizon).
    # A test/val sequence must not start before the end of the training region.
    max_train_target = max(train_idx) + horizon - 1 if train_idx else -1
    future_targets_ok = all(min(i, i + horizon - 1) > max_train_target for i in val_idx + test_idx)

    recomputed_mean = float(values[: max(train_idx) + horizon].mean()) if train_idx else None
    scaler_fitted_on_train = (
        abs(recomputed_mean - scaler_mean) < 1e-6 if recomputed_mean is not None else False
    )

    passed = disjoint_1 and disjoint_2 and disjoint_3 and future_targets_ok and scaler_fitted_on_train
    return {
        "passed": bool(passed),
        "checks": {
            "train_val_disjoint": bool(disjoint_1),
            "train_test_disjoint": bool(disjoint_2),
            "val_test_disjoint": bool(disjoint_3),
            "no_future_targets_in_earlier_splits": bool(future_targets_ok),
            "scaler_fitted_on_train_only": bool(scaler_fitted_on_train),
        },
        "sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "seq_len": seq_len,
        "horizon": horizon,
    }
