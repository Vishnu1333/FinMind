"""End-to-end ML pipeline tests (Phase 19).

Covers: dataset loading, preprocessing / sequence generation, leakage-free
splits, training reproducibility, model artifact loading + checksum
verification, artifact-backed inference (forecast / anomaly / risk),
metrics correctness, the prediction API route, and the Gemini controlled
tool that consumes the forecasting service.

All assertions run against the REAL processed datasets and REAL trained
artifacts produced by the ml.training.* pipelines.
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import models  # noqa: E402
from backend.app.db import SessionLocal  # noqa: E402
from backend.app.services.provisioning_service import ProvisioningService  # noqa: E402
from backend.app.agent import tools as agent_tools  # noqa: E402
from ml.evaluation.metrics import classification_metrics, regression_metrics  # noqa: E402
from ml.preprocessing.timeseries import (  # noqa: E402
    apply_zscore,
    chronological_split,
    fit_zscore,
    invert_zscore,
    leakage_audit,
    make_sequences,
    moving_average_forecast,
    naive_last_value,
)
from ml.forecasting.lstm_model import RecurrentForecaster  # noqa: E402
from ml.inference import anomaly_inference, forecast_inference  # noqa: E402
from ml.registry.registry import ModelRegistry  # noqa: E402


@pytest.fixture()
def session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture()
def provisioned(session):
    business = models.Business(name="ML Pipeline Test Sdn Bhd")
    session.add(business)
    session.flush()
    user = models.User(email="ml_pipeline_test@finmind.com", hashed_password="x", business_id=business.id)
    session.add(user)
    session.commit()
    session.refresh(business)
    session.refresh(user)
    ProvisioningService(session).provision(business, user)
    return user


# ---------------------------------------------------------------------------
# 1. Datasets: processed files + honest manifests
# ---------------------------------------------------------------------------
class TestDatasets:
    def test_processed_forecasting_series_exist(self):
        import pandas as pd

        for path in (
            "data/processed/forecasting/online_retail_daily.csv",
            "data/processed/forecasting/superstore_daily.csv",
        ):
            assert os.path.exists(path), f"missing processed series: {path}"
            df = pd.read_csv(path, parse_dates=["date"])
            assert len(df) > 300
            assert df.iloc[:, 1].notna().all()
            assert df["date"].is_monotonic_increasing  # chronological order

    def test_processed_fraud_features_exist(self):
        for path in ("data/processed/fraud/paysim_features.npz", "data/processed/fraud/ulb_features.npz"):
            assert os.path.exists(path), f"missing processed features: {path}"
            data = np.load(path)
            assert data["X"].shape[0] == data["y"].shape[0]
            assert data["y"].ndim == 1

    def test_manifests_match_reality(self):
        import json

        expected = {
            "online_retail_manifest.json": "data/raw/online_retail/online-retail.zip",
            "superstore_manifest.json": "data/raw/superstore/Sample - Superstore.csv",
            "ealaxi_paysim1_manifest.json": "data/raw/ealaxi_paysim1/paysim1.zip",
            "mlg-ulb_creditcardfraud_manifest.json": "data/raw/mlg-ulb_creditcardfraud/creditcardfraud.zip",
        }
        for manifest_file, raw_file in expected.items():
            with open(os.path.join("data/manifests", manifest_file)) as f:
                m = json.load(f)
            assert m["rows"] > 0 and m["columns"] > 0
            assert m["license"] and "license" in m["license"].lower() or m["license"]
            assert m["target"] and m["features"]
            # the checksum in the manifest must match the file on disk
            import hashlib

            h = hashlib.sha256()
            with open(raw_file, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            assert m["checksum"][raw_file] == h.hexdigest()

    def test_unavailable_datasets_are_not_claimed_as_downloaded(self):
        import json

        for slug in ("eaubet_paysim1", "ieee-cis-fraud-detection", "m5-forecasting-accuracy"):
            with open(f"data/manifests/{slug}_manifest.json") as f:
                m = json.load(f)
            assert m.get("status") == "not_available"
            assert m.get("status_reason")


# ---------------------------------------------------------------------------
# 2. Preprocessing: sequences, chronological splits, train-only scaling
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_make_sequences_shape_and_alignment(self):
        v = np.arange(100, dtype=float)
        X, y, starts = make_sequences(v, seq_len=14, horizon=1)
        assert X.shape == (86, 14, 1)
        assert y.shape == (86, 1)
        # X[0] must be v[0:14], target must be v[14]
        np.testing.assert_array_equal(X[0, :, 0], v[:14])
        assert y[0, 0] == v[14]
        assert starts[0] == 14

    def test_chronological_split_no_overlap_ordered(self):
        train, val, test = chronological_split(200, 0.7, 0.15)
        assert max(train) < min(val) < max(val) < min(test)
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)
        assert len(train) + len(val) + len(test) == 200

    def test_chronological_split_refuses_bad_fractions(self):
        with pytest.raises(ValueError):
            chronological_split(100, 0.95, 0.05)  # leaves no test set
        with pytest.raises(ValueError):
            chronological_split(1, 0.5, 0.2)  # empty segments
        with pytest.raises(ValueError):
            chronological_split(100, 0.0, 0.15)  # invalid fraction

    def test_scaler_train_only_and_leakage_audit(self):
        rng = np.random.RandomState(0)
        values = rng.normal(10, 3, 400).cumsum()
        X, y, starts = make_sequences(values, 14, 1)
        train_idx, val_idx, test_idx = chronological_split(len(X), 0.7, 0.15)
        mu, sd = fit_zscore(values[: starts[train_idx[-1]] + 1])
        scaled = apply_zscore(values, mu, sd)
        # audit verifies disjointness, ordering AND that the scaler stats
        # equal the training-segment statistics
        audit = leakage_audit(
            values,
            [int(i) for i in starts[train_idx]],
            [int(i) for i in starts[val_idx]],
            [int(i) for i in starts[test_idx]],
            14, 1, mu, sd,
        )
        assert audit["passed"] is True

        # a deliberately-leaky scaler (fitted on ALL data) must FAIL the audit
        bad_mu = float(values.mean())
        audit_bad = leakage_audit(
            values,
            [int(i) for i in starts[train_idx]],
            [int(i) for i in starts[val_idx]],
            [int(i) for i in starts[test_idx]],
            14, 1, bad_mu, sd,
        )
        assert audit_bad["checks"]["scaler_fitted_on_train_only"] is False

    def test_inverse_zscore_roundtrip(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        mu, sd = fit_zscore(v)
        np.testing.assert_allclose(invert_zscore(apply_zscore(v, mu, sd), mu, sd), v)

    def test_baselines(self):
        ctx = np.array([1.0, 2.0, 3.0, 10.0])
        np.testing.assert_array_equal(naive_last_value(ctx, 3), [10.0, 10.0, 10.0])
        np.testing.assert_array_equal(
            moving_average_forecast(ctx, 2, window=2), [6.5, 6.5]  # mean(3, 10)
        )


# ---------------------------------------------------------------------------
# 3. Training: real BPTT on the real Online Retail series + reproducibility
# ---------------------------------------------------------------------------
class TestTraining:
    @pytest.fixture()
    def retail_series(self):
        import pandas as pd

        df = pd.read_csv("data/processed/forecasting/online_retail_daily.csv")
        return df.iloc[:, 1].to_numpy(np.float64)

    def test_all_three_cells_learn_on_real_data(self, retail_series):
        mu, sd = fit_zscore(retail_series[: int(len(retail_series) * 0.85)])
        scaled = apply_zscore(retail_series, mu, sd)
        X, y, _ = make_sequences(scaled, 14, 1)
        for cell in ("rnn", "lstm", "gru"):
            model = RecurrentForecaster(cell=cell, hidden_size=16, seq_len=14, lr=0.03, seed=42)
            losses = model.fit(X[:200], y[:200], epochs=10)
            assert losses[-1] < losses[0], f"{cell} did not learn on real data"
            fc = model.forecast(scaled[-14:], 5)
            assert fc.shape == (5,) and np.all(np.isfinite(fc))

    def test_seed_reproducibility(self, retail_series):
        mu, sd = fit_zscore(retail_series[: int(len(retail_series) * 0.85)])
        scaled = apply_zscore(retail_series, mu, sd)
        X, y, _ = make_sequences(scaled, 14, 1)
        runs = []
        for _ in range(2):
            model = RecurrentForecaster(cell="gru", hidden_size=8, seq_len=14, lr=0.03, seed=7)
            runs.append(model.fit(X[:100], y[:100], epochs=5))
        np.testing.assert_array_equal(runs[0], runs[1])


# ---------------------------------------------------------------------------
# 4. Model artifacts + registry
# ---------------------------------------------------------------------------
class TestModelArtifacts:
    def test_registry_has_all_expected_models(self):
        registry = ModelRegistry()
        for name in ("finmind_rnn", "finmind_lstm", "finmind_gru", "finmind_ar14",
                     "finmind_anomaly", "finmind_fraud_paysim", "finmind_fraud_ulb",
                     "finmind_risk_credit", "finmind_risk_german"):
            assert registry.get(name) is not None, f"{name} missing from registry"
            assert registry.get(name)["status"] == "validated"

    def test_selected_forecasting_model_exists(self):
        registry = ModelRegistry()
        selected = [e for e in registry.all().values()
                    if e.get("task") == "forecasting" and e.get("selected")]
        assert len(selected) == 1
        assert selected[0]["metrics"]["pooled_recursive_30d_scaled"]["mae"] > 0

    def test_artifact_checksums_match_registry(self):
        registry = ModelRegistry()
        broken = [
            name for name, e in registry.all().items()
            if e.get("artifact_path") and not registry.validate_artifact(name)
        ]
        assert broken == [], f"checksum mismatch for: {broken}"

    def test_forecasting_artifact_is_deterministic_on_reload(self):
        forecast_inference.reset_manager()
        import pandas as pd

        df = pd.read_csv("data/processed/forecasting/superstore_daily.csv")
        values = df.iloc[:, 1].to_numpy(np.float64)[:120]
        out1 = forecast_inference.forecast_series(values, 10)
        forecast_inference.reset_manager()
        out2 = forecast_inference.forecast_series(values, 10)
        assert out1["predictions"] == out2["predictions"]  # deterministic
        assert len(out1["predictions"]) == 10

    def test_inference_rejects_short_series(self):
        forecast_inference.reset_manager()
        with pytest.raises(ValueError):
            forecast_inference.forecast_series([1.0, 2.0, 3.0], 5)


# ---------------------------------------------------------------------------
# 5. Metrics correctness
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_regression_metrics_known_values(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 290.0])
        m = regression_metrics(y_true, y_pred)
        assert abs(m["mae"] - 10.0) < 1e-6
        assert abs(m["rmse"] - 10.0) < 1e-6
        # MAPE = mean(|err|/|true|) * 100
        expected_mape = np.mean([10 / 100, 10 / 200, 10 / 300]) * 100
        assert abs(m["mape_pct"] - expected_mape) < 1e-3
        # ss_res = 300, ss_tot = 20000 -> R^2 = 0.985
        assert abs(m["r2"] - 0.985) < 1e-3

    def test_classification_metrics(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        perfect = np.array([0.05, 0.05, 0.05, 0.05, 0.95, 0.95])
        m = classification_metrics(y_true, perfect)
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
        assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0

    def test_mape_exclusions_are_reported_not_hidden(self):
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([5.0, 110.0, 190.0])
        m = regression_metrics(y_true, y_pred)
        assert m["mape_excluded_points"] == 1  # zero-denominator point excluded


# ---------------------------------------------------------------------------
# 6. Inference + API + Gemini tool integration
# ---------------------------------------------------------------------------
class TestInferenceIntegration:
    def test_forecast_service_artifact_backed_no_training(self, session, provisioned):
        from backend.app.services.forecast_service import ForecastService

        service = ForecastService(session, provisioned.business_id)
        result = service.forecast(days=14)
        assert result["currency"] == "MYR"
        assert len(result["forecast"]) == 14
        # provenance: registry-selected artifact, checksum verified
        assert result["model"]["artifact_checksum_verified"] is True
        assert result["model"]["selected"] in ("LSTM", "GRU", "RNN", "AR(14)")
        # the artifact provides real offline holdout metrics
        assert result["model"]["holdout_mae_scaled"] > 0

    def test_prediction_api_route(self, session, provisioned):
        from fastapi.testclient import TestClient

        from backend.app.main import app
        from backend.app import db as database
        from backend.app.dependencies import get_current_user

        def _override_db():
            yield session

        def _override_user():
            return provisioned

        app.dependency_overrides[database.get_db] = _override_db
        app.dependency_overrides[get_current_user] = _override_user
        try:
            client = TestClient(app)
            resp = client.get("/predictions/forecast?days=7")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["currency"] == "MYR"
            assert len(body["forecast"]) == 7
            assert all("predicted_cashflow" in p for p in body["forecast"])
            assert body["model"]["artifact"].startswith("finmind_")
            assert body["model"]["trained_on_days"] >= 60

            # anomalies endpoint (artifact-backed service)
            resp2 = client.get("/predictions/anomalies?top_n=5")
            assert resp2.status_code == 200
            body2 = resp2.json()
            assert body2["available"] is True
            assert body2["model"]["artifact"] == "finmind_anomaly"
            assert body2["model"]["trained_on"] >= 40
        finally:
            app.dependency_overrides.clear()

    def test_gemini_tool_get_forecast_contract(self, session, provisioned):
        # The controlled tool Gemini calls (Phase 12) — same function the
        # Gemini function-calling loop and the deterministic engine use.
        result = agent_tools.get_forecast(session, provisioned.business_id, {"days": 14})
        assert result["success"] is True
        assert result["kind"] == "PREDICTION"
        assert result["source"] == "forecast_service"
        assert result["data"]["days"] == 14
        assert len(result["data"]["series"]) == 14
        assert result["data"]["model"]["selected"] in ("LSTM", "GRU", "RNN", "AR(14)")
        # text lines contain real model output only
        assert any("RM" in line for line in result["text_lines"])

    def test_anomaly_tool_reports_artifact_provenance(self, session, provisioned):
        result = agent_tools.get_anomalies(session, provisioned.business_id, {"limit": 10})
        assert result["success"] is True
        assert result["data"]["model"]["artifact"] == "finmind_anomaly"
        assert result["data"]["model"]["artifact_checksum_verified"] is True

    def test_risk_artifact_loads_and_scores(self):
        # credit-risk artifacts are validated but not wired to the live API
        # (no credit-feature schema for businesses) — they must still load.
        import joblib

        model = joblib.load("data/models/finmind_risk_credit/model.pkl")
        scaler = joblib.load("data/models/finmind_risk_credit/scaler.pkl")
        data = np.load("data/processed/risk/uci_credit_default.npz")
        row = data["X"][0:1].astype(np.float64)
        proba = float(model.predict_proba(scaler.transform(row))[0, 1])
        assert 0.0 <= proba <= 1.0

    def test_anomaly_artifact_score_is_real(self):
        anomaly_inference.reset_manager()
        out = anomaly_inference.score_features(
            [np.log1p(120.0), 0.0, 1.0, 0.0, 1.0, 0.0]
        )
        assert isinstance(out["score"], float)
        assert -1.0 < out["score"] < 0.0  # IsolationForest score range
