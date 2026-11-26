"""Tests for the per-business forecasting service and the NumPy RNN models."""

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
from backend.app.services.forecast_service import ForecastService, InsufficientDataError  # noqa: E402
from backend.app.services.anomaly_service import AnomalyService  # noqa: E402
from ml.forecasting.lstm_model import RecurrentForecaster  # noqa: E402
from ml.forecasting.ar_model import ARForecaster  # noqa: E402


@pytest.fixture()
def session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture()
def provisioned(session):
    business = models.Business(name="Forecast Test Sdn Bhd")
    session.add(business)
    session.flush()
    user = models.User(email="forecast_test@finmind.com", hashed_password="x", business_id=business.id)
    session.add(user)
    session.commit()
    session.refresh(business)
    session.refresh(user)
    ProvisioningService(session).provision(business, user)
    return user


class TestNumPyRNN:
    def test_lstm_learns(self):
        rng = np.random.RandomState(0)
        t = np.arange(200)
        series = 500 + 2.2 * t + 180 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 40, 200)
        scaled = (series - series.mean()) / series.std()
        X = np.stack([scaled[i:i + 14] for i in range(len(scaled) - 14)])[:, :, None]
        y = scaled[14:].reshape(-1, 1)
        model = RecurrentForecaster(cell="lstm", hidden_size=16, seq_len=14, lr=0.03, seed=42)
        losses = model.fit(X, y, epochs=50)
        assert losses[-1] < losses[0] * 0.5  # loss must meaningfully decrease

    def test_gru_learns(self):
        rng = np.random.RandomState(0)
        t = np.arange(200)
        series = 500 + 2.2 * t + 180 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 40, 200)
        scaled = (series - series.mean()) / series.std()
        X = np.stack([scaled[i:i + 14] for i in range(len(scaled) - 14)])[:, :, None]
        y = scaled[14:].reshape(-1, 1)
        model = RecurrentForecaster(cell="gru", hidden_size=16, seq_len=14, lr=0.03, seed=42)
        losses = model.fit(X, y, epochs=50)
        assert losses[-1] < losses[0] * 0.5

    def test_forecast_shape(self):
        rng = np.random.RandomState(1)
        series = np.cumsum(rng.randn(120)) + 500
        model = RecurrentForecaster(cell="lstm", seq_len=14, seed=3)
        model.fit(
            np.stack([series[i:i + 14] for i in range(len(series) - 14)])[:, :, None],
            series[14:].reshape(-1, 1), epochs=20,
        )
        fc = model.forecast(series[-14:], 14)
        assert fc.shape == (14,)
        assert np.all(np.isfinite(fc))

    def test_ar_forecaster(self):
        rng = np.random.RandomState(2)
        ar_true = ARForecaster(lags=14)
        # AR(2) process with known coefficients
        x = np.zeros(200)
        for i in range(2, 200):
            x[i] = 0.6 * x[i - 1] - 0.3 * x[i - 2] + rng.normal(0, 0.1)
        ar_true.fit(x[:160])
        fc = ar_true.forecast(x[:160], 10)
        assert fc.shape == (10,)
        # Predictions should be close to the true continuation
        assert np.abs(fc - x[160:170]).mean() < 0.5

    def test_invalid_cell_rejected(self):
        with pytest.raises(ValueError):
            RecurrentForecaster(cell="transformer")


class TestForecastService:
    def test_forecast_on_provisioned_business(self, session, provisioned):
        service = ForecastService(session, provisioned.business_id)
        result = service.forecast(days=14)
        assert len(result["forecast"]) == 14
        # The selected model is whatever the offline evaluation promoted
        # (registry-selected artifact) — one of the real candidates.
        assert result["model"]["selected"] in ("LSTM", "GRU", "RNN", "AR(14)")
        assert result["model"]["trained_on_days"] >= 60
        # Artifact-backed inference provenance must be present and honest.
        assert result["model"]["artifact"].startswith("finmind_")
        assert result["model"]["artifact_checksum_verified"] is True
        assert result["model"]["business_context_days"] >= 100
        assert isinstance(result["model"]["candidates"], dict)
        assert len(result["model"]["candidates"]) >= 4  # rnn/lstm/gru + baselines
        # Every forecast value is a finite float produced by the model
        for point in result["forecast"]:
            assert isinstance(point["predicted_cashflow"], float)
        # Cumulative series must be a running sum
        cum = 0.0
        for point in result["forecast"]:
            cum += point["predicted_cashflow"]
            assert abs(point["cumulative"] - round(cum, 2)) < 0.011

    def test_insufficient_data_raises(self, session):
        business = models.Business(name="Empty Biz Sdn Bhd")
        session.add(business)
        session.flush()
        user = models.User(email="empty@finmind.com", hashed_password="x", business_id=business.id)
        session.add(user)
        session.commit()
        service = ForecastService(session, business.id)
        with pytest.raises(InsufficientDataError):
            service.forecast(days=7)

    def test_daily_series_is_real(self, session, provisioned):
        service = ForecastService(session, provisioned.business_id)
        series = service._daily_net_series()
        assert len(series) >= 100  # ~180 days of provisioned history
        assert all("date" in p and "net" in p for p in series)


class TestAnomalyService:
    def test_detect_finds_planted_anomalies(self, session, provisioned):
        service = AnomalyService(session, provisioned.business_id)
        result = service.detect()
        assert result["available"] is True
        assert result["model"]["trained_on"] >= 40
        # The provisioned history contains a planted RM 8,950 equipment purchase
        amounts = [a["amount"] for a in result["anomalies"]]
        assert any(a >= 8000 for a in amounts), f"planted anomaly not flagged; got {amounts[:10]}"

    def test_min_amount_filter(self, session, provisioned):
        service = AnomalyService(session, provisioned.business_id)
        result = service.detect(min_amount=500)
        assert all(a["amount"] >= 500 for a in result["anomalies"])

    def test_insufficient_history(self, session):
        business = models.Business(name="Tiny Biz Sdn Bhd")
        session.add(business)
        session.flush()
        user = models.User(email="tiny@finmind.com", hashed_password="x", business_id=business.id)
        session.add(user)
        session.commit()
        service = AnomalyService(session, business.id)
        result = service.detect()
        assert result["available"] is False
        assert "at least" in result["reason"].lower()
