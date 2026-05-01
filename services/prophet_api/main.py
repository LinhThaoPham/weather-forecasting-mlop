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

from src.config.constants import DEFAULT_CITY, model_filename

# Models storage
prophet_hourly = None
prophet_daily = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models", "current")


def _resolve_model_path(mode: str) -> str | None:
    """Resolve Prophet model path with default city, then fallback to any city artifact."""
    preferred_name = model_filename("prophet", mode, DEFAULT_CITY)
    preferred_path = os.path.join(MODELS_DIR, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path

    if not os.path.isdir(MODELS_DIR):
        return None

    prefix = f"prophet_{mode}_"
    candidates = sorted(
        f for f in os.listdir(MODELS_DIR)
        if f.startswith(prefix) and f.endswith(".json")
    )
    if not candidates:
        return None

    fallback_path = os.path.join(MODELS_DIR, candidates[0])
    print(
        f"⚠ Preferred {mode} model '{preferred_name}' not found. "
        f"Using fallback '{candidates[0]}'."
    )
    return fallback_path


def _load_models():
    """Load Prophet models from disk."""
    global prophet_hourly, prophet_daily

    try:
        prophet_hourly = None
        prophet_daily = None

        hourly_path = _resolve_model_path("hourly")
        if hourly_path and os.path.exists(hourly_path):
            with open(hourly_path, "r", encoding="utf-8") as f:
                prophet_hourly = model_from_json(json.load(f))
            print(f"✓ Prophet Hourly model loaded from {hourly_path}")
        else:
            print("⚠ Prophet Hourly model not found in models/current")

        daily_path = _resolve_model_path("daily")
        if daily_path and os.path.exists(daily_path):
            with open(daily_path, "r", encoding="utf-8") as f:
                prophet_daily = model_from_json(json.load(f))
            print(f"✓ Prophet Daily model loaded from {daily_path}")
        else:
            print("⚠ Prophet Daily model not found in models/current")

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
    data: List[Dict[str, Any]]  # [{ds, humidity, cloud_cover}, ...]
    mode: str = "hourly"        # "hourly" | "daily"


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
        "hourly_loaded": prophet_hourly is not None,
        "daily_loaded": prophet_daily is not None
    }


@app.post("/reload")
def reload_models():
    """Hot-reload models from disk (called by retrain pipeline)."""
    try:
        _load_models()
        return {
            "status": "success",
            "message": "Models reloaded",
            "hourly_loaded": prophet_hourly is not None,
            "daily_loaded": prophet_daily is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.post("/forecast", response_model=ProphetForecastResponse)
def forecast(request: ProphetForecastRequest):
    """Predict temperatures using Prophet model.

    Expects data with columns: ds, humidity, cloud_cover
    Returns: temperature predictions (yhat)
    """
    try:
        model = prophet_hourly if request.mode == "hourly" else prophet_daily

        if model is None:
            raise HTTPException(status_code=503, detail=f"{request.mode} model not loaded")

        df = pd.DataFrame(request.data)
        df['ds'] = pd.to_datetime(df['ds'])

        forecast_result = model.predict(df[['ds', 'humidity', 'cloud_cover']])

        return ProphetForecastResponse(
            status="success",
            mode=request.mode,
            predictions=forecast_result['yhat'].tolist()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


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
        "message": "Prophet API đang chạy",
        "hourly_loaded": prophet_hourly is not None,
        "daily_loaded": prophet_daily is not None,
        "endpoints": {
            "forecast": "/forecast (POST - recommended)",
            "predict_hourly": "/predict_hourly (POST - legacy)",
            "predict_daily": "/predict_daily (POST - legacy)",
            "reload": "/reload (POST)",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
