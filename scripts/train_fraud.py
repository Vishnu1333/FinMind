"""DEPRECATED shim — use the new fraud training pipeline instead.

The old script had two leakage defects (fixed in the new pipeline):

1. `preprocess_pipeline` fitted the StandardScaler on the FULL dataset
   BEFORE the random split (test statistics leaked into training).
2. A random split of highly duplicated transaction records allowed the
   same record to appear in both training and evaluation.

It also trained an IsolationForest on ULB PCA features (V1..V28) which do
not exist for FinMind business transactions, so the artifact could never
be used by the live service.

Real fraud/anomaly training now lives in:

    ml/training/train_fraud_models.py

Run it with:
    ./backend/venv/bin/python -m ml.training.train_fraud_models
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.training.train_fraud_models import main  # noqa: F401,E402

if __name__ == "__main__":
    main()
