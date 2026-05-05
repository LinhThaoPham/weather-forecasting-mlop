# services/prophet_api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any
from prophet.serialize import model_from_json
import json
import pandas as pd
import os

from src.config.constants import DEFAULT_CITY, PROPHET_EXTRA_TARGETS, model_filename
from src.config.cities import CITIES
from src.config.gcs_storage import sync_models_from_gcs

# Per-city models: {"hanoi": model, "hcm": model, ...}
prophet_hourly_models: dict = {}
prophet_daily_models: dict = {}
# Extra target models: {"humidity": {"hanoi": model, ...}, ...}
prophet_extra_models: dict = {}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models", "current")


def _load_models():
    """Load Prophet models for ALL cities from disk."""
    global prophet_hourly_models, prophet_daily_models

    try:
        downloaded = sync_models_from_gcs(MODELS_DIR)
        if downloaded:
            print(f"☁ Downloaded {downloaded} model artifacts from GCS")

        prophet_hourly_models = {}
        prophet_daily_models = {}

        for city_id in CITIES:
            # Hourly
            hourly_name = model_filename("prophet", "hourly", city_id)
            hourly_path = os.path.join(MODELS_DIR, hourly_name)
            if os.path.exists(hourly_path):
                with open(hourly_path, "r", encoding="utf-8") as f:
                    prophet_hourly_models[city_id] = model_from_json(json.load(f))
                print(f"  ✓ Prophet Hourly [{city_id}] loaded")

            # Daily
            daily_name = model_filename("prophet", "daily", city_id)
            daily_path = os.path.join(MODELS_DIR, daily_name)
            if os.path.exists(daily_path):
                with open(daily_path, "r", encoding="utf-8") as f:
                    prophet_daily_models[city_id] = model_from_json(json.load(f))
                print(f"  ✓ Prophet Daily  [{city_id}] loaded")

        print(f"✓ Prophet loaded: {len(prophet_hourly_models)} hourly, {len(prophet_daily_models)} daily")

        # Load extra-target models (humidity, wind_speed, cloud_cover)
        global prophet_extra_models
        prophet_extra_models = {}
        for target in PROPHET_EXTRA_TARGETS:
            prophet_extra_models[target] = {}
            for city_id in CITIES:
                fname = model_filename("prophet", "hourly", city_id, target)
                fpath = os.path.join(MODELS_DIR, fname)
                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        prophet_extra_models[target][city_id] = model_from_json(json.load(f))
                    print(f"  ✓ Prophet {target} [{city_id}] loaded")

        extra_count = sum(len(v) for v in prophet_extra_models.values())
        print(f"✓ Prophet extra targets loaded: {extra_count} models")

    except Exception as e:
        print(f"❌ Error loading Prophet models: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    print("Prophet API shutting down")


app = FastAPI(title="Prophet API - Weather Việt Nam", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== Request/Response Models ==================

class ProphetForecastRequest(BaseModel):
    data: List[Dict[str, Any]]  # [{"ds": "2026-05-03 12:00:00"}, ...]
    mode: str = "hourly"        # "hourly" | "daily"
    city: str = "hanoi"         # per-city model selection


class ProphetForecastResponse(BaseModel):
    status: str
    mode: str
    predictions: List[float]


# ================== Endpoints ==================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "prophet_api",
        "hourly_cities": list(prophet_hourly_models.keys()),
        "daily_cities": list(prophet_daily_models.keys()),
    }


@app.post("/reload")
def reload_models():
    """Hot-reload models from disk (called by retrain pipeline)."""
    try:
        _load_models()
        return {
            "status": "success",
            "message": "Models reloaded",
            "hourly_cities": list(prophet_hourly_models.keys()),
            "daily_cities": list(prophet_daily_models.keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.post("/forecast", response_model=ProphetForecastResponse)
def forecast(request: ProphetForecastRequest):
    """Predict temperatures using Prophet model (pure time-series).

    Expects data with column: ds (timestamp only, no exogenous variables)
    Returns: temperature predictions (yhat)
    """
    try:
        models = prophet_hourly_models if request.mode == "hourly" else prophet_daily_models
        model = models.get(request.city)

        # Fallback: try default city if requested city not found
        if model is None:
            model = models.get(DEFAULT_CITY)
        if model is None:
            available = list(models.keys())
            raise HTTPException(
                status_code=503,
                detail=f"No {request.mode} model for '{request.city}'. Available: {available}"
            )

        df = pd.DataFrame(request.data)
        df['ds'] = pd.to_datetime(df['ds'])

        forecast_result = model.predict(df[['ds']])

        return {
            "status": "success",
            "mode": request.mode,
            "predictions": forecast_result['yhat'].tolist(),
            "confidence_lower": forecast_result['yhat_lower'].tolist(),
            "confidence_upper": forecast_result['yhat_upper'].tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


class ProphetMultiVarRequest(BaseModel):
    data: List[Dict[str, Any]]
    city: str = "hanoi"


@app.post("/forecast_multi_var")
def forecast_multi_var(request: ProphetMultiVarRequest):
    """Predict ALL weather variables (humidity, wind, cloud) for a city.

    Returns dict: {"humidity": [...], "wind_speed": [...], "cloud_cover": [...]}
    """
    try:
        df = pd.DataFrame(request.data)
        df['ds'] = pd.to_datetime(df['ds'])

        result = {}
        for target in PROPHET_EXTRA_TARGETS:
            models_for_target = prophet_extra_models.get(target, {})
            model = models_for_target.get(request.city)
            if model is None:
                model = models_for_target.get(DEFAULT_CITY)
            if model is not None:
                forecast_result = model.predict(df[['ds']])
                preds = forecast_result['yhat'].tolist()
                # Clamp values to reasonable ranges
                if target == "humidity":
                    preds = [max(0, min(100, p)) for p in preds]
                elif target == "cloud_cover":
                    preds = [max(0, min(100, p)) for p in preds]
                elif target == "wind_speed":
                    preds = [max(0, p) for p in preds]
                elif target == "precipitation":
                    preds = [max(0, round(p, 1)) for p in preds]
                result[target] = preds
            else:
                result[target] = None

        return {"status": "success", "city": request.city, "predictions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-var prediction error: {str(e)}")


# ================== Legacy Endpoints ==================

@app.post("/predict_hourly")
def predict_hourly(data: List[Dict[str, Any]]):
    if prophet_hourly is None:
        raise HTTPException(status_code=503, detail="Hourly model not loaded")
    df = pd.DataFrame(data)
    forecast_result = prophet_hourly.predict(df)
    return {"predictions": forecast_result['yhat'].tolist()}


@app.post("/predict_daily")
def predict_daily(data: List[Dict[str, Any]]):
    if prophet_daily is None:
        raise HTTPException(status_code=503, detail="Daily model not loaded")
    df = pd.DataFrame(data)
    forecast_result = prophet_daily.predict(df)
    return {"predictions": forecast_result['yhat'].tolist()}


@app.get("/")
def root():
    return {
        "message": "Prophet API — Per-City Models",
        "hourly_cities": list(prophet_hourly_models.keys()),
        "daily_cities": list(prophet_daily_models.keys()),
        "endpoints": {
            "forecast": "/forecast (POST - per-city temperature)",
            "forecast_multi_var": "/forecast_multi_var (POST - humidity/wind/cloud)",
            "predict_hourly": "/predict_hourly (POST - legacy)",
            "predict_daily": "/predict_daily (POST - legacy)",
            "reload": "/reload (POST)",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
