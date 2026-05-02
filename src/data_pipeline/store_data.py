"""Fetch weather data from APIs → store in SQLite.

Functions:
    store_historical(city_id, days)  — Open-Meteo Archive → weather_historical
    store_forecast(city_id, days)    — Open-Meteo Forecast → weather_forecast
    store_current(city_id)           — OWM → weather_current
"""
import time

import requests

from src.config.cities import CITIES, get_city_coords
from src.config.db import get_connection, init_db
from src.config.gcp import USE_BIGQUERY
from src.config.settings import (
    API_TIMEOUT,
    COLUMN_MAP,
    FORECAST_DAYS_DEFAULT,
    HISTORICAL_DAYS,
    HOURLY_PARAMS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    OWM_API_KEY,
    OWM_BASE_URL,
    OWM_TIMEOUT,
    TIMEZONE,
)
from src.data_pipeline.bigquery_storage import append_historical_rows


def _parse_open_meteo_hourly(data: dict) -> list[dict]:
    """Convert Open-Meteo hourly response into list of row dicts."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    # Map API field names → DB column names
    field_map = {
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

    rows = []
    for i, ts in enumerate(times):
        row = {"timestamp": ts}
        for api_key, db_col in field_map.items():
            values = hourly.get(api_key, [])
            row[db_col] = values[i] if i < len(values) else None
        rows.append(row)
    return rows


_HIST_INSERT = """INSERT OR IGNORE INTO weather_historical
    (city_id, timestamp, temperature, humidity, cloud_cover,
     apparent_temp, precipitation, rain, weather_code,
     pressure, wind_speed, wind_direction, wind_gusts, dewpoint)
    VALUES (:city_id, :timestamp, :temperature, :humidity, :cloud_cover,
     :apparent_temp, :precipitation, :rain, :weather_code,
     :pressure, :wind_speed, :wind_direction, :wind_gusts, :dewpoint)"""

_FC_INSERT = """INSERT OR REPLACE INTO weather_forecast
    (city_id, timestamp, temperature, humidity, cloud_cover,
     apparent_temp, precipitation, rain, weather_code,
     pressure, wind_speed, wind_direction, wind_gusts, dewpoint, forecast_days)
    VALUES (:city_id, :timestamp, :temperature, :humidity, :cloud_cover,
     :apparent_temp, :precipitation, :rain, :weather_code,
     :pressure, :wind_speed, :wind_direction, :wind_gusts, :dewpoint, :days)"""


def store_historical(city_id: str, days: int = HISTORICAL_DAYS) -> int:
    """Fetch 2-year hourly history from Open-Meteo Archive → SQLite.

    Returns number of rows inserted.
    """
    from datetime import datetime, timedelta

    coords = get_city_coords(city_id)
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_PARAMS,
        "timezone": TIMEZONE,
    }

    print(f"  📡 Fetching {city_id} historical ({start_date} → {end_date})...")
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=API_TIMEOUT)
    resp.raise_for_status()

    rows = _parse_open_meteo_hourly(resp.json())

    with get_connection() as conn:
        conn.executemany(_HIST_INSERT, [{"city_id": city_id, **r} for r in rows])
        inserted = conn.total_changes

    if USE_BIGQUERY:
        append_historical_rows(city_id, rows)

    print(f"  ✓ {city_id}: {len(rows)} rows fetched, stored to DB")
    return inserted


def store_forecast(city_id: str, days: int = FORECAST_DAYS_DEFAULT) -> int:
    """Fetch forecast from Open-Meteo → SQLite.

    Returns number of rows inserted.
    """
    coords = get_city_coords(city_id)

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": HOURLY_PARAMS,
        "forecast_days": days,
        "timezone": TIMEZONE,
    }

    print(f"  📡 Fetching {city_id} forecast ({days} days)...")
    resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=API_TIMEOUT)
    resp.raise_for_status()

    rows = _parse_open_meteo_hourly(resp.json())

    with get_connection() as conn:
        conn.executemany(_FC_INSERT, [{"city_id": city_id, "days": days, **r} for r in rows])

    print(f"  ✓ {city_id}: {len(rows)} forecast rows stored")
    return len(rows)


def store_current(city_id: str) -> dict | None:
    """Fetch current weather from OWM → SQLite.

    Returns the weather dict or None on failure.
    """
    if not OWM_API_KEY:
        print(f"  ⚠ OWM key not configured, skipping {city_id}")
        return None

    coords = get_city_coords(city_id)
    url = (
        f"{OWM_BASE_URL}"
        f"?lat={coords['lat']}&lon={coords['lon']}"
        f"&appid={OWM_API_KEY}&units=metric"
    )

    try:
        resp = requests.get(url, timeout=OWM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  ⚠ OWM fetch failed for {city_id}: {e}")
        return None

    weather = data.get("weather", [{}])[0]
    main = data.get("main", {})
    wind = data.get("wind", {})
    clouds = data.get("clouds", {})

    row = {
        "city_id": city_id,
        "temperature": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "cloud_cover": clouds.get("all"),
        "wind_speed": round(wind.get("speed", 0) * 3.6, 1),
        "weather_code": weather.get("id", 0),
        "weather_desc": weather.get("description", ""),
        "visibility": round(data.get("visibility", 0) / 1000, 1),
    }

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO weather_current
               (city_id, temperature, feels_like, humidity, pressure,
                cloud_cover, wind_speed, weather_code, weather_desc, visibility)
               VALUES (:city_id, :temperature, :feels_like, :humidity, :pressure,
                :cloud_cover, :wind_speed, :weather_code, :weather_desc, :visibility)""",
            row,
        )

    print(f"  ✓ {city_id}: current {row['temperature']}°C stored")
    return row


def seed_all_cities(days: int = HISTORICAL_DAYS, delay: float = 0.5) -> None:
    """Fetch & store data for ALL 6 cities.

    Args:
        days: historical days to fetch (default 730 = 2 years)
        delay: seconds between API calls to avoid rate limiting
    """
    init_db()
    city_ids = list(CITIES.keys())
    total = len(city_ids)

    print(f"\n{'='*60}")
    print(f"🌍 Seeding {total} cities × {days} days")
    print(f"{'='*60}\n")

    for i, city_id in enumerate(city_ids, 1):
        print(f"[{i}/{total}] {city_id}")

        store_historical(city_id, days=days)
        time.sleep(delay)

        store_forecast(city_id)
        time.sleep(delay)

        store_current(city_id)
        time.sleep(delay)

        print()

    # Summary
    with get_connection() as conn:
        for table in ["weather_historical", "weather_forecast", "weather_current"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"📊 {table}: {count:,} rows")

    print(f"\n{'='*60}")
    print("✅ Seeding complete!")
    print(f"{'='*60}\n")
