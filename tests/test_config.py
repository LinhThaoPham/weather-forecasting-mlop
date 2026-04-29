"""Tests for config modules — settings, cities, constants, db."""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Settings ──

class TestSettings:
    def test_project_root_exists(self):
        from src.config.settings import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

    def test_db_path_has_extension(self):
        from src.config.settings import DB_PATH
        assert DB_PATH.endswith(".db")

    def test_api_urls_are_strings(self):
        from src.config.settings import OPEN_METEO_ARCHIVE_URL
        assert isinstance(OPEN_METEO_ARCHIVE_URL, str)
        assert OPEN_METEO_ARCHIVE_URL.startswith("http")

    def test_service_urls_default(self):
        from src.config.settings import DATA_API_URL, FORECAST_API_URL
        assert "localhost" in DATA_API_URL or "127.0.0.1" in DATA_API_URL
        assert "localhost" in FORECAST_API_URL or "127.0.0.1" in FORECAST_API_URL


# ── Cities ──

class TestCities:
    def test_cities_not_empty(self):
        from src.config.cities import CITIES
        assert len(CITIES) > 0

    def test_each_city_has_required_fields(self):
        from src.config.cities import CITIES
        required_keys = {"name", "lat", "lon"}
        for city_id, info in CITIES.items():
            assert required_keys.issubset(info.keys()), f"{city_id} missing keys"

    def test_hanoi_exists(self):
        from src.config.cities import CITIES
        assert "hanoi" in CITIES

    def test_coordinates_valid_range(self):
        from src.config.cities import CITIES
        for city_id, info in CITIES.items():
            assert -90 <= info["lat"] <= 90, f"{city_id} lat out of range"
            assert -180 <= info["lon"] <= 180, f"{city_id} lon out of range"


# ── Constants ──

class TestConstants:
    def test_lookback_positive(self):
        from src.config.constants import DAILY_LOOKBACK
        assert DAILY_LOOKBACK > 0

    def test_forecast_horizon_positive(self):
        from src.config.constants import DAILY_HORIZON
        assert DAILY_HORIZON > 0

    def test_prophet_weight_valid(self):
        from src.config.constants import PROPHET_WEIGHT
        assert 0 <= PROPHET_WEIGHT <= 1


# ── Database ──

class TestDatabase:
    def test_get_connection_sqlite(self):
        from src.config.db import get_connection
        with get_connection() as conn:
            assert conn is not None

    def test_init_db_creates_tables(self):
        from src.config.db import init_db, get_connection
        init_db()
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            assert "weather_historical" in tables
            assert "weather_forecast" in tables
            assert "weather_current" in tables
