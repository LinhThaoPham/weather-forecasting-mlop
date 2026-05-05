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
    """Lấy dự báo AI từ bảng weather_ai_predictions.

    Dữ liệu này được tạo bởi daily_pipeline (hoặc forecast_api/predict).
    Dashboard hiển thị kết quả DỰ BÁO CỦA AI, không phải của Open-Meteo.
    """
    try:
        coords = get_city_coords(city)
        hours = days * 24

        with get_connection() as conn:
            rows = conn.execute(
                """SELECT target_time, predicted_temp, predicted_humidity,
                          predicted_wind_speed, predicted_cloud_cover,
                          model_version, created_at
                   FROM weather_ai_predictions
                   WHERE city_id = ?
                   ORDER BY target_time
                   LIMIT ?""",
                (city, hours),
            ).fetchall()

        if rows:
            data = [dict(r) for r in rows]
            return {
                "city": coords["name"],
                "lat": coords["lat"],
                "lon": coords["lon"],
                "forecast_days": days,
                "source": "ai_model",
                "model_version": data[0].get("model_version", "unknown") if data else None,
                "records_count": len(data),
                "data": {
                    "hourly": {
                        "time": [d["target_time"] for d in data],
                        "temperature_2m": [d["predicted_temp"] for d in data],
                        "humidity": [d["predicted_humidity"] for d in data],
                        "wind_speed": [d["predicted_wind_speed"] for d in data],
                        "cloud_cover": [d["predicted_cloud_cover"] for d in data],
                    }
                },
            }

        # Fallback: no AI predictions yet → return empty with message
        return {
            "city": coords["name"],
            "lat": coords["lat"],
            "lon": coords["lon"],
            "forecast_days": days,
            "source": "none",
            "message": "No AI predictions available yet. Run daily_pipeline.py first.",
            "data": {"hourly": {"time": [], "temperature_2m": [], "humidity": [], "wind_speed": [], "cloud_cover": []}},
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


@app.get("/model/performance")
def get_performance_history():
    """Serve daily MAE performance history for monitoring dashboard."""
    import json
    perf_path = os.path.join(MODELS_DIR, "performance_history.json")
    if not os.path.exists(perf_path):
        return {"entries": [], "message": "No performance history yet. Run daily_pipeline.py first."}

    with open(perf_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {"entries": _sanitize_json(entries), "count": len(entries)}


@app.get("/model/alerts")
def get_model_alerts():
    """Return active alerts based on model performance monitoring.

    Checks:
      - Sustained drift (7+ days MAE > threshold)
      - Cooldown status (days since last retrain)
      - Recent performance trend
    """
    import json
    from src.config.constants import DRIFT_THRESHOLD, SUSTAINED_DRIFT_DAYS, RETRAIN_COOLDOWN_DAYS

    alerts = []

    # Check performance history for sustained drift
    perf_path = os.path.join(MODELS_DIR, "performance_history.json")
    if os.path.exists(perf_path):
        with open(perf_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        if history:
            # Last entry status
            latest = history[-1]
            if latest.get("exceeded_threshold"):
                alerts.append({
                    "level": "warning",
                    "type": "threshold_exceeded",
                    "message": f"MAE ({latest['avg_mae']}°C) exceeded threshold ({DRIFT_THRESHOLD}°C) on {latest['date']}",
                })

            # Check sustained drift
            if len(history) >= SUSTAINED_DRIFT_DAYS:
                recent = history[-SUSTAINED_DRIFT_DAYS:]
                consecutive = sum(1 for e in recent if e.get("exceeded_threshold", False))
                if consecutive >= SUSTAINED_DRIFT_DAYS:
                    alerts.append({
                        "level": "critical",
                        "type": "sustained_drift",
                        "message": f"MAE exceeded threshold for {SUSTAINED_DRIFT_DAYS} consecutive days! Retrain triggered.",
                    })
                elif consecutive > 0:
                    alerts.append({
                        "level": "info",
                        "type": "intermittent_drift",
                        "message": f"{consecutive}/{SUSTAINED_DRIFT_DAYS} recent days exceeded threshold.",
                    })

    # Check cooldown status from registry
    registry_path = os.path.join(MODELS_DIR, "registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            raw = f.read()
        raw = raw.replace("Infinity", "null").replace("NaN", "null")
        registry = json.loads(raw)

        models = registry.get("models", [])
        if models:
            last_accepted = None
            for m in reversed(models):
                if m.get("decision") == "accept":
                    last_accepted = m
                    break
            if last_accepted:
                trained_at = last_accepted.get("trained_at", "")
                try:
                    trained_date = datetime.fromisoformat(trained_at.split("+")[0])
                    days_since = (datetime.now() - trained_date).days
                    alerts.append({
                        "level": "info",
                        "type": "cooldown_status",
                        "message": f"Last retrain: {days_since} days ago (cooldown: {RETRAIN_COOLDOWN_DAYS}d)",
                        "days_since_retrain": days_since,
                        "cooldown_active": days_since < RETRAIN_COOLDOWN_DAYS,
                    })
                except (ValueError, TypeError):
                    pass

    if not alerts:
        alerts.append({"level": "ok", "type": "healthy", "message": "All systems healthy. No alerts."})

    return {"alerts": alerts, "timestamp": datetime.now().isoformat()}


@app.get("/model/explainability")
def get_model_explainability():
    """Return model explainability: feature importance and per-model performance.

    Shows which model (Prophet vs LSTM) performs better per city,
    current ensemble weights, and training configuration.
    """
    import json
    from src.config.constants import (
        PROPHET_EXTRA_TARGETS, WEATHER_TARGETS,
        SUSTAINED_DRIFT_DAYS, RETRAIN_COOLDOWN_DAYS,
        RETRAIN_INTERVAL_DAYS, DRIFT_THRESHOLD,
    )

    result = {
        "ensemble_strategy": "Dynamic weights (inverse MAE)",
        "targets": {
            "temperature": "Prophet + LSTM ensemble (dynamic weights)",
            **{t: "Prophet only" for t in PROPHET_EXTRA_TARGETS},
        },
        "retrain_policy": {
            "seasonal_interval_days": RETRAIN_INTERVAL_DAYS,
            "sustained_drift_days": SUSTAINED_DRIFT_DAYS,
            "cooldown_days": RETRAIN_COOLDOWN_DAYS,
            "drift_threshold_celsius": DRIFT_THRESHOLD,
        },
        "feature_engineering": [
            {"name": "temp_lag_1..24", "type": "lag", "description": "Historical temperature at 1,2,3,6,12,24 hours ago"},
            {"name": "temp_rolling_7d", "type": "trend", "description": "7-day rolling average temperature"},
            {"name": "temp_amplitude", "type": "stability", "description": "Max-min temperature in 24h window"},
            {"name": "pressure_change_24h", "type": "weather_signal", "description": "Pressure change in 24h (cold front indicator)"},
            {"name": "dewpoint_spread", "type": "weather_signal", "description": "Temperature - Dewpoint (rain/fog predictor)"},
            {"name": "day_of_week", "type": "temporal", "description": "Day of week (0-6)"},
            {"name": "month", "type": "temporal", "description": "Month (1-12, seasonal signal)"},
            {"name": "hour", "type": "temporal", "description": "Hour of day (0-23, diurnal cycle)"},
        ],
    }

    # Add per-model performance from registry
    registry_path = os.path.join(MODELS_DIR, "registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            raw = f.read()
        raw = raw.replace("Infinity", "null").replace("NaN", "null")
        registry = json.loads(raw)

        models = registry.get("models", [])
        latest = None
        for m in reversed(models):
            if m.get("decision") == "accept":
                latest = m
                break

        if latest:
            metrics = latest.get("metrics", {})
            model_performance = {}
            for key, val in metrics.items():
                if isinstance(val, dict) and "mae" in val:
                    model_performance[key] = _sanitize_json(val)

            result["current_version"] = latest.get("version")
            result["trained_at"] = latest.get("trained_at")
            result["model_performance"] = model_performance

            # Calculate dynamic weights
            import numpy as np
            prophet_maes = []
            for k, v in model_performance.items():
                if k.startswith("prophet_hourly") and v.get("mae"):
                    prophet_maes.append(v["mae"])
            lstm_mae = model_performance.get("lstm_hourly", {}).get("mae")

            if prophet_maes and lstm_mae:
                avg_prophet = np.mean(prophet_maes)
                inv_p = 1.0 / avg_prophet
                inv_l = 1.0 / lstm_mae
                pw = round(max(0.3, min(0.7, inv_p / (inv_p + inv_l))), 2)
                result["dynamic_weights"] = {
                    "prophet": pw,
                    "lstm": round(1 - pw, 2),
                    "reason": f"Prophet avg MAE={avg_prophet:.2f}, LSTM MAE={lstm_mae:.2f}"
                }

    return result


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
            "model_performance": "/model/performance",
            "model_alerts": "/model/alerts",
            "model_explainability": "/model/explainability",
            "health": "/health",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

