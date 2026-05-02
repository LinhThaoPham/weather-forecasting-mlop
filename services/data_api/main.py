# services/data_api/main.py
"""Data API — serves weather data from SQLite + live API fallback."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

from src.config.cities import CITIES, get_city_coords
from src.config.db import get_connection, init_db
from src.config.gcp import USE_GCS
from src.config.gcs_storage import download_file
from src.config.settings import (
    API_TIMEOUT,
    HOURLY_PARAMS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    OWM_API_KEY,
    OWM_BASE_URL,
    OWM_TIMEOUT,
    TIMEZONE,
)

# OWM Weather ID → WMO-like code for dashboard compatibility
OWM_TO_WMO = {
    range(200, 233): 95,   # Thunderstorm → WMO 95
    range(300, 322): 51,   # Drizzle → WMO 51
    range(500, 532): 61,   # Rain → WMO 61
    range(600, 623): 71,   # Snow → WMO 71
    range(701, 782): 45,   # Atmosphere (fog) → WMO 45
    range(800, 801): 0,    # Clear → WMO 0
    range(801, 805): 2,    # Clouds → WMO 2
}


def owm_code_to_wmo(owm_id: int) -> int:
    """Convert OWM weather ID to WMO-like code for dashboard."""
    for code_range, wmo_code in OWM_TO_WMO.items():
        if owm_id in code_range:
            return wmo_code
    return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("✓ Data API started (SQLite backed)")
    yield
    print("Data API shutting down")


app = FastAPI(title="Data API - Weather Việt Nam", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    with get_connection() as conn:
        hist_count = conn.execute("SELECT COUNT(*) FROM weather_historical").fetchone()[0]
        fc_count = conn.execute("SELECT COUNT(*) FROM weather_forecast").fetchone()[0]
        cur_count = conn.execute("SELECT COUNT(*) FROM weather_current").fetchone()[0]

    return {
        "status": "healthy",
        "service": "data_api",
        "db_rows": {
            "historical": hist_count,
            "forecast": fc_count,
            "current": cur_count,
        },
    }


@app.get("/cities")
def get_cities():
    """Get list of supported cities."""
    return {
        "cities": [
            {"id": key, "name": v["name"], "lat": v["lat"], "lon": v["lon"], "country": v["country"]}
            for key, v in CITIES.items()
        ]
    }


@app.get("/historical")
def get_historical(days: int = 1000, city: str = "hanoi"):
    """Lấy dữ liệu lịch sử — ưu tiên từ SQLite, fallback Open-Meteo API."""
    # Try DB first
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT timestamp as time, temperature as temperature_2m,
                      humidity as relative_humidity_2m, cloud_cover
               FROM weather_historical
               WHERE city_id = ?
               ORDER BY timestamp""",
            (city,),
        ).fetchall()

    if rows:
        city_info = get_city_coords(city)
        data = [dict(r) for r in rows]
        return {
            "city": city_info["name"],
            "source": "database",
            "records_count": len(data),
            "data": data,
        }

    # Fallback: call API directly
    try:
        coords = get_city_coords(city)
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

        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df['city'] = coords["name"]

        return {
            "city": coords["name"],
            "source": "api_fallback",
            "records_count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/current")
def get_current(city: str = "hanoi"):
    """Lấy dữ liệu realtime từ OWM → lưu DB → trả về."""
    try:
        coords = get_city_coords(city)

        if not OWM_API_KEY:
            return {"error": "OPENWEATHER_API_KEY not configured"}

        url = (
            f"{OWM_BASE_URL}"
            f"?lat={coords['lat']}&lon={coords['lon']}"
            f"&appid={OWM_API_KEY}&units=metric"
        )

        resp = requests.get(url, timeout=OWM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        weather = data.get("weather", [{}])[0]
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})

        result = {
            "city": coords["name"],
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "cloud_cover": clouds.get("all"),
            "wind_speed": round(wind.get("speed", 0) * 3.6, 1),
            "weather_code": owm_code_to_wmo(weather.get("id", 0)),
            "weather_desc": weather.get("description", ""),
            "visibility": round(data.get("visibility", 0) / 1000, 1),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "lat": coords["lat"],
            "lon": coords["lon"],
        }

        # Store to DB
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO weather_current
                   (city_id, temperature, feels_like, humidity, pressure,
                    cloud_cover, wind_speed, weather_code, weather_desc, visibility)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (city, result["temperature"], result["feels_like"],
                 result["humidity"], result["pressure"], result["cloud_cover"],
                 result["wind_speed"], result["weather_code"],
                 result["weather_desc"], result["visibility"]),
            )

        return result

    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast")
def get_forecast(days: int = 3, city: str = "hanoi"):
    """Lấy dữ liệu forecast từ Open-Meteo → lưu DB → trả về."""
    try:
        coords = get_city_coords(city)

        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": HOURLY_PARAMS,
            "forecast_days": days,
            "timezone": TIMEZONE,
        }

        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        api_data = resp.json()

        # Store forecast to DB
        hourly = api_data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humids = hourly.get("relative_humidity_2m", [])
        clouds = hourly.get("cloud_cover", [])

        with get_connection() as conn:
            for i, ts in enumerate(times):
                conn.execute(
                    """INSERT OR REPLACE INTO weather_forecast
                       (city_id, timestamp, temperature, humidity, cloud_cover, forecast_days)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (city, ts,
                     temps[i] if i < len(temps) else None,
                     humids[i] if i < len(humids) else None,
                     clouds[i] if i < len(clouds) else None,
                     days),
                )

        return {
            "city": coords["name"],
            "lat": coords["lat"],
            "lon": coords["lon"],
            "forecast_days": days,
            "data": api_data,
        }

    except Exception as e:
        return {"error": str(e)}


# ============== Model Monitoring ==============

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))


def _sanitize_json(obj):
    """Replace NaN/Infinity with None for JSON compliance."""
    import math
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


@app.get("/model/registry")
def get_model_registry():
    """Serve model registry (metrics across versions)."""
    registry_path = os.path.join(MODELS_DIR, "registry.json")
    if USE_GCS:
        download_file("models/registry.json", registry_path)
    if not os.path.exists(registry_path):
        return {"error": "registry.json not found"}

    import json
    with open(registry_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Handle NaN/Infinity in JSON (non-standard but written by Python)
    raw = raw.replace("Infinity", "null").replace("NaN", "null")
    data = json.loads(raw)
    return _sanitize_json(data)


@app.get("/model/history")
def get_training_history():
    """Serve training history (epoch-level loss curves)."""
    history_path = os.path.join(MODELS_DIR, "training_history.json")
    if USE_GCS:
        download_file("models/training_history.json", history_path)
    if not os.path.exists(history_path):
        return {"entries": [], "message": "No training history yet"}

    import json
    with open(history_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {"entries": _sanitize_json(entries)}


@app.get("/")
def root():
    return {
        "message": "Data API đang chạy (SQLite backed)",
        "endpoints": {
            "cities": "/cities",
            "historical": "/historical?city=hanoi&days=1000",
            "current": "/current?city=hanoi",
            "forecast": "/forecast?city=hanoi&days=3",
            "model_registry": "/model/registry",
            "model_history": "/model/history",
            "health": "/health",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

