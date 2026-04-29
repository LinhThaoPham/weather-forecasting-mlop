"""Centralized Settings — Single source of truth for all config.

All services import from here. Values come from:
  1. Environment variables (.env) — highest priority
  2. Defaults below — fallback
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project Root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── External API URLs ──
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# ── API Keys ──
OWM_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ── Internal Service URLs ──
DATA_API_URL = os.getenv("DATA_API_URL", "http://127.0.0.1:8001")
FORECAST_API_URL = os.getenv("FORECAST_API_URL", "http://127.0.0.1:8000")
PROPHET_API_URL = os.getenv("PROPHET_API_URL", "http://127.0.0.1:8003")
LSTM_API_URL = os.getenv("LSTM_API_URL", "http://127.0.0.1:8004")

# ── Timeouts (seconds) ──
API_TIMEOUT = 30
OWM_TIMEOUT = 10
SERVICE_CALL_TIMEOUT = 60

# ── Data Params ──
HOURLY_PARAMS = (
    "temperature_2m,relative_humidity_2m,cloud_cover,"
    "apparent_temperature,precipitation,rain,weather_code,"
    "pressure_msl,wind_speed_10m,wind_direction_10m,"
    "wind_gusts_10m,dewpoint_2m"
)
DAILY_PARAMS = (
    "weather_code,temperature_2m_max,temperature_2m_min,"
    "apparent_temperature_max,apparent_temperature_min,"
    "precipitation_sum,precipitation_hours,"
    "wind_speed_10m_max,wind_gusts_10m_max,"
    "wind_direction_10m_dominant"
)
TIMEZONE = "Asia/Ho_Chi_Minh"
FORECAST_DAYS_DEFAULT = 3
HISTORICAL_DAYS = 730  # 2 years

# ── Database ──
DB_PATH = str(PROJECT_ROOT / "data" / "weather_forecast.db")

# ── Column Mapping (Open-Meteo → internal) ──
COLUMN_MAP = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "cloud_cover": "cloud_cover",
    "apparent_temperature": "apparent_temp",
    "precipitation": "precipitation",
    "rain": "rain",
    "weather_code": "weather_code",
    "pressure_msl": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "wind_gusts_10m": "wind_gusts",
    "dewpoint_2m": "dewpoint",
}
