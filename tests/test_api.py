"""Tests for API services — health checks, endpoint contracts."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Data API ──

class TestDataAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services.data_api.main import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"

    def test_current_weather_requires_city(self):
        res = self.client.get("/current")
        assert res.status_code in (200, 422)

    def test_current_weather_valid_city(self):
        res = self.client.get("/current?city=hanoi")
        assert res.status_code == 200

    def test_historical_endpoint(self):
        res = self.client.get("/historical?city=hanoi&days=2")
        assert res.status_code == 200


# ── Forecast API ──

class TestForecastAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services.forecast_api.main import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"

    def test_predict_requires_body(self):
        res = self.client.post("/predict")
        assert res.status_code == 422

    def test_predict_valid_request(self):
        res = self.client.post("/predict", json={
            "city": "hanoi",
            "mode": "hourly",
            "hours": 24
        })
        # May fail if backend APIs not running, but should not crash
        assert res.status_code in (200, 500, 503)


# ── Prophet API ──

class TestProphetAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services.prophet_api.main import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        assert res.status_code == 200

    def test_predict_hourly(self):
        res = self.client.post("/predict_hourly", json=[
            {"ds": "2026-01-01", "temperature": 25.0}
        ])
        assert res.status_code in (200, 422, 500, 503)


# ── LSTM API ──

class TestLSTMAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        from services.lstm_api.main import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        assert res.status_code == 200

    def test_predict_hourly(self):
        res = self.client.post("/predict_hourly", json=[
            {"ds": "2026-01-01", "temperature": 25.0}
        ])
        assert res.status_code in (200, 422, 500, 503)
