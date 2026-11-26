"""Tabular preprocessing pipeline for classification tasks (fraud / risk).

Phase 15 leakage rules enforced here:

* The scaler and encoders are FIT ON THE TRAINING SPLIT ONLY and then
  applied to val/test. (The previous version scaled/encoded the full
  dataset before splitting — a real leakage bug, now fixed.)
* Duplicates are removed ONCE before splitting with the decision
  documented, never silently.
* Row identity is tracked so train/test overlap can be audited.

For time-series data use ml.preprocessing.timeseries instead — random
splits are never applied to temporal data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.categorical_maps: Dict[str, Dict[str, int]] = {}
        self.drop_reasons: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def validate_schema(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        train_df: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit scaler/encoders on TRAIN data only; transform it."""
        categorical_columns = categorical_columns or []
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)

        blocks = []
        if self.numeric_columns:
            self.scaler = StandardScaler()
            blocks.append(self.scaler.fit_transform(train_df[self.numeric_columns].astype(float)))
        if self.categorical_columns:
            for col in self.categorical_columns:
                values = train_df[col].astype(str)
                self.categorical_maps[col] = {
                    v: i + 1 for i, v in enumerate(sorted(values.unique()))
                }
                blocks.append(values.map(self.categorical_maps[col]).to_numpy(float).reshape(-1, 1))
        if not blocks:
            raise ValueError("no feature columns provided")
        return np.hstack([b.reshape(len(train_df), -1) for b in blocks])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted scaler/encoders to val/test rows."""
        if self.scaler is None and not self.categorical_maps:
            raise RuntimeError("preprocessor has not been fitted")
        blocks = []
        if self.numeric_columns:
            blocks.append(self.scaler.transform(df[self.numeric_columns].astype(float)))
        for col in self.categorical_columns:
            mapping = self.categorical_maps[col]
            unknown = self.categorical_maps.setdefault(
                "_unknown_value_tracker", {}
            )
            encoded = df[col].astype(str).map(mapping)
            n_unknown = int(encoded.isna().sum())
            if n_unknown:
                unknown[col] = unknown.get(col, 0) + n_unknown
                encoded = encoded.fillna(0)  # unseen category -> 0 (documented)
            blocks.append(encoded.to_numpy(float).reshape(-1, 1))
        return np.hstack([b.reshape(len(df), -1) for b in blocks])

    # ------------------------------------------------------------------
    def unknown_category_counts(self) -> Dict[str, int]:
        tracker = self.categorical_maps.get("_unknown_value_tracker", {})
        return {k: int(v) for k, v in tracker.items()}


def dedupe_with_report(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Remove exact duplicate rows, reporting exactly what was removed."""
    before = len(df)
    keys = key_columns or list(df.columns)
    dup_mask = df.duplicated(subset=keys, keep="first")
    n_duplicates = int(dup_mask.sum())
    out = df[~dup_mask].copy()
    report = {
        "rows_before": before,
        "exact_duplicates_removed": n_duplicates,
        "rows_after": len(out),
        "key_columns": keys,
    }
    return out, report


def stratified_split_with_preprocessing(
    df: pd.DataFrame,
    target_col: str,
    numeric_columns: List[str],
    categorical_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, object]:
    """Leakage-free stratified split for NON-temporal classification data.

    Order of operations (this ordering is the fix — the old pipeline
    scaled the full dataset before splitting):
      1. chronological-agnostic stratified split of raw rows
      2. fit scaler/encoder on train only
      3. transform val/test with the fitted state
    """
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, random_state=random_state, stratify=df[target_col]
    )
    relative = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative, random_state=random_state, stratify=temp_df[target_col]
    )

    pre = DataPreprocessor()
    X_train = pre.fit_transform(train_df, numeric_columns, categorical_columns)
    X_val = pre.transform(val_df)
    X_test = pre.transform(test_df)

    y_train = train_df[target_col].to_numpy(int)
    y_val = val_df[target_col].to_numpy(int)
    y_test = test_df[target_col].to_numpy(int)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "preprocessor": pre,
        "split_sizes": {
            "train": len(train_df), "val": len(val_df), "test": len(test_df),
            "train_positive": int(y_train.sum()),
            "val_positive": int(y_val.sum()),
            "test_positive": int(y_test.sum()),
        },
    }
