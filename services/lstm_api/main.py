# services/lstm_api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np
import os
import joblib
from typing import List

from src.models_logic.lstm_model import LSTMWeatherModel

# Models storage
lstm_hourly = None
lstm_daily = None
scaler_hourly = None
scaler_daily = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models", "current")


def _load_model_file(filename: str) -> str:
    """Get model file path — from GCS or local filesystem."""
    try:
        from src.config.gcs_storage import download_model
        return download_model(filename, MODELS_DIR)
    except (ImportError, FileNotFoundError):
        local_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(local_path):
            return local_path
        return ""


def _load_models():
    """Load LSTM models and scalers from GCS or local disk."""
    global lstm_hourly, lstm_daily, scaler_hourly, scaler_daily

    try:
        # Load hourly model
        hourly_path = _load_model_file("lstm_hourly_hanoi.h5")
        if hourly_path:
            lstm_hourly_obj = LSTMWeatherModel(lookback_window=24, forecast_horizon=72)
            lstm_hourly_obj.load(hourly_path)
            lstm_hourly = lstm_hourly_obj.model

            scaler_path = _load_model_file("lstm_hourly_all_scaler.pkl")
            if scaler_path:
                scaler_hourly = joblib.load(scaler_path)
            print("✓ LSTM Hourly model loaded")
        else:
            print("⚠ LSTM Hourly model not found")

        # Load daily model
        daily_path = _load_model_file("lstm_daily_hanoi.h5")
        if daily_path:
            lstm_daily_obj = LSTMWeatherModel(lookback_window=7, forecast_horizon=7)
            lstm_daily_obj.load(daily_path)
            lstm_daily = lstm_daily_obj.model

            scaler_path = _load_model_file("lstm_daily_all_scaler.pkl")
            if scaler_path:
                scaler_daily = joblib.load(scaler_path)
            print("✓ LSTM Daily model loaded")
        else:
            print("⚠ LSTM Daily model not found")

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    print("LSTM API shutting down")


app = FastAPI(title="LSTM Weather Forecast API - Hà Nội", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== Request/Response Models ==================

class LSTMPredictRequest(BaseModel):
    sequences: List[List[float]]


class LSTMPredictResponse(BaseModel):
    status: str
    mode: str
    predictions: List[List[float]]


class LSTMForecastRequest(BaseModel):
    temperatures: List[float]  # Raw °C values
    mode: str = "hourly"       # "hourly" | "daily"


class LSTMForecastResponse(BaseModel):
    status: str
    mode: str
    predictions: List[float]   # Raw °C predictions


# ================== Endpoints ==================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "lstm_api",
        "hourly_loaded": lstm_hourly is not None,
        "daily_loaded": lstm_daily is not None
    }


@app.post("/reload")
def reload_models():
    """Hot-reload models from disk (called by retrain pipeline)."""
    try:
        _load_models()
        return {
            "status": "success",
            "message": "Models reloaded",
            "hourly_loaded": lstm_hourly is not None,
            "daily_loaded": lstm_daily is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.post("/forecast", response_model=LSTMForecastResponse)
def forecast(request: LSTMForecastRequest):
    """High-level endpoint: raw temps in → raw temp predictions out.
    Handles normalization/denormalization internally.
    """
    try:
        if request.mode == "hourly":
            model = lstm_hourly
            scaler = scaler_hourly
            window_size = 24
        else:
            model = lstm_daily
            scaler = scaler_daily
            window_size = 7

        if model is None:
            raise HTTPException(status_code=503, detail=f"{request.mode} model not loaded")
        if scaler is None:
            raise HTTPException(status_code=503, detail=f"{request.mode} scaler not loaded")

        temps = np.array(request.temperatures, dtype=np.float32)

        # Normalize
        normalized = scaler.transform(temps.reshape(-1, 1)).flatten()

        # Create sequence (last window_size values)
        sequence = normalized[-window_size:].tolist() if len(normalized) >= window_size else normalized.tolist()
        while len(sequence) < window_size:
            sequence.insert(0, sequence[0] if sequence else 0)

        # Predict
        input_array = np.array([sequence], dtype=np.float32).reshape(1, window_size, 1)
        predictions_normalized = model.predict(input_array)
        predictions_flat = predictions_normalized.reshape(-1)

        # Denormalize
        predictions = scaler.inverse_transform(predictions_flat.reshape(-1, 1)).flatten()

        return LSTMForecastResponse(
            status="success",
            mode=request.mode,
            predictions=predictions.tolist()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ================== Legacy Endpoints (backward-compatible) ==================

@app.post("/predict_hourly", response_model=LSTMPredictResponse)
def predict_hourly(request: LSTMPredictRequest):
    """Predict hourly temperature (72 hours ahead). Legacy endpoint."""
    try:
        if lstm_hourly is None:
            raise HTTPException(status_code=503, detail="Hourly model not loaded")

        sequences = np.array(request.sequences, dtype=np.float32)
        sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
        predictions = lstm_hourly.predict(sequences)
        predictions_list = predictions.reshape((predictions.shape[0], -1)).tolist()

        return LSTMPredictResponse(status="success", mode="hourly", predictions=predictions_list)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_daily", response_model=LSTMPredictResponse)
def predict_daily(request: LSTMPredictRequest):
    """Predict daily temperature (7 days ahead). Legacy endpoint."""
    try:
        if lstm_daily is None:
            raise HTTPException(status_code=503, detail="Daily model not loaded")

        sequences = np.array(request.sequences, dtype=np.float32)
        sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
        predictions = lstm_daily.predict(sequences)
        predictions_list = predictions.reshape((predictions.shape[0], -1)).tolist()

        return LSTMPredictResponse(status="success", mode="daily", predictions=predictions_list)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "LSTM Weather API đang chạy",
        "endpoints": {
            "forecast": "/forecast (POST - recommended)",
            "hourly": "/predict_hourly (POST - legacy)",
            "daily": "/predict_daily (POST - legacy)",
            "reload": "/reload (POST)",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
