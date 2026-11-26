"""DEPRECATED shim — use the new forecasting training pipeline instead.

The old script trained an "LSTM forecaster" on the UCI Default of Credit
Card Clients dataset by slicing its BILL_AMT1 column into a pseudo
time-series. That dataset is a tabular client snapshot with NO temporal
ordering, so the exercise was invalid for forecasting (its own holdout
MAPE was 276.69%). The artifact it produced (lstm_forecast_v1) is marked
deprecated in the model registry with the audit reason.

Real forecasting training now lives in:

    ml/training/train_forecast_models.py

Run it with:
    ./backend/venv/bin/python -m ml.training.train_forecast_models
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.training.train_forecast_models import main  # noqa: F401,E402

if __name__ == "__main__":
    main()
