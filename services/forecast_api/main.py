# services/forecast_api/main.py
"""
Forecast API — Pure Orchestrator.
Gathers data from Open-Meteo + OWM, calls Prophet API + LSTM API, blends results.
Does NOT load any models directly.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
import requests
import numpy as np

from src.config.cities import CITIES, get_city_coords
from src.config.settings import (
    API_TIMEOUT,
    FORECAST_DAYS_DEFAULT,
    HOURLY_PARAMS,
    LSTM_API_URL,
    OPEN_METEO_FORECAST_URL,
    OWM_API_KEY,
    OWM_BASE_URL,
    OWM_TIMEOUT,
    PROPHET_API_URL,
    SERVICE_CALL_TIMEOUT,
    TIMEZONE,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✓ Forecast API started (Pure Orchestrator)")
    print(f"  PROPHET_API_URL: {PROPHET_API_URL}")
    print(f"  LSTM_API_URL: {LSTM_API_URL}")
    yield
    print("Forecast API shutting down")


app = FastAPI(
    title="Weather ML API - Việt Nam (Prophet + LSTM)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    city: str = "hanoi"
    days: int = 7
    hours: int = 72
    mode: str = "hourly"


# ============== DATA FETCHING ==============

def get_forecast_data(hours=72, city="hanoi"):
    """Fetch forecast data from Open-Meteo API."""
    try:
        coords = get_city_coords(city)

        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": HOURLY_PARAMS,
            "forecast_days": FORECAST_DAYS_DEFAULT,
            "timezone": TIMEZONE,
        }

        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data['hourly'])
        df['ds'] = pd.to_datetime(df['time'])
        df = df.rename(columns={
            "temperature_2m": "y",
            "relative_humidity_2m": "humidity",
            "cloud_cover": "cloud_cover"
        })
        df = df.drop(columns=['time'])
        return df.head(hours)

    except Exception as e:
        print(f"Error fetching forecast for {city}: {e}")
        return None


def get_owm_current(city="hanoi"):
    """Fetch current weather from OpenWeatherMap to supplement gaps."""
    if not OWM_API_KEY:
        return None

    try:
        coords = get_city_coords(city)
        url = (
            f"{OWM_BASE_URL}"
            f"?lat={coords['lat']}&lon={coords['lon']}"
            f"&appid={OWM_API_KEY}&units=metric"
        )

        resp = requests.get(url, timeout=OWM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "cloud_cover": data["clouds"]["all"]
        }
    except Exception as e:
        print(f"⚠ OWM fallback failed: {e}")
        return None


def supplement_with_owm(df, city="hanoi"):
    """Add OWM current data to fill Open-Meteo gaps."""
    owm = get_owm_current(city)
    if owm is None or df is None:
        return df

    now = pd.Timestamp.now().floor('h')
    # Only supplement if current hour is missing from forecast
    if now not in df['ds'].values:
        new_row = pd.DataFrame([{
            'ds': now,
            'y': owm['temperature'],
            'humidity': owm['humidity'],
            'cloud_cover': owm['cloud_cover']
        }])
        df = pd.concat([new_row, df], ignore_index=True)
        df = df.sort_values('ds').reset_index(drop=True)

    return df


# ============== SERVICE CALLS ==============

def call_prophet_api(df, mode="hourly"):
    """Call Prophet API for predictions."""
    try:
        data = df[['ds', 'humidity', 'cloud_cover']].copy()
        data['ds'] = data['ds'].astype(str)

        response = requests.post(
            f"{PROPHET_API_URL}/forecast",
            json={"data": data.to_dict(orient="records"), "mode": mode},
            timeout=SERVICE_CALL_TIMEOUT,
        )
        result = response.json()

        if result.get("status") == "success":
            return np.array(result["predictions"])
        else:
            print(f"⚠ Prophet API error: {result}")
            return None

    except Exception as e:
        print(f"⚠ Prophet API call failed: {e}")
        return None


def call_lstm_api(temperatures, mode="hourly"):
    """Call LSTM API for predictions."""
    try:
        response = requests.post(
            f"{LSTM_API_URL}/forecast",
            json={"temperatures": temperatures.tolist(), "mode": mode},
            timeout=SERVICE_CALL_TIMEOUT,
        )
        result = response.json()

        if result.get("status") == "success":
            return np.array(result["predictions"])
        else:
            print(f"⚠ LSTM API error: {result}")
            return None

    except Exception as e:
        print(f"⚠ LSTM API call failed: {e}")
        return None


# ============== ENSEMBLE ==============

def ensemble_predictions(prophet_pred, lstm_pred, prophet_weight=0.6, lstm_weight=0.4):
    """Blend Prophet and LSTM predictions (0.6 Prophet + 0.4 LSTM)."""
    if lstm_pred is None:
        return prophet_pred
    if prophet_pred is None:
        return lstm_pred

    min_len = min(len(prophet_pred), len(lstm_pred))
    return prophet_weight * prophet_pred[:min_len] + lstm_weight * lstm_pred[:min_len]


# ============== MAIN PREDICTION ==============

def predict_weather(hours=72, days=7, mode="hourly", city="hanoi"):
    """Main prediction: Open-Meteo data → Prophet API + LSTM API → Ensemble."""
    try:
        # 1. Get forecast data from Open-Meteo
        df = get_forecast_data(hours=hours, city=city)
        if df is None or len(df) == 0:
            return None

        # 2. Supplement with OWM current weather
        df = supplement_with_owm(df, city=city)

        # 3. Call Prophet API
        prophet_preds = call_prophet_api(df, mode=mode)
        if prophet_preds is None:
            print("⚠ Prophet failed, using Open-Meteo forecast as fallback")
            prophet_preds = df['y'].values

        # 4. Call LSTM API
        lstm_preds = call_lstm_api(df['y'].values, mode=mode)

        # 5. Ensemble
        final_preds = ensemble_predictions(prophet_preds, lstm_preds)

        # Format output
        df_out = df[['ds']].copy()
        df_out['prophet_pred'] = prophet_preds[:len(df_out)]
        df_out['lstm_pred'] = lstm_preds[:len(df_out)] if lstm_preds is not None else np.nan
        df_out['final_pred'] = final_preds[:len(df_out)]

        # Aggregate to daily if needed
        if mode == "daily":
            df_out['date'] = pd.to_datetime(df_out['ds']).dt.date
            df_out = df_out.groupby('date').agg({
                'prophet_pred': 'mean',
                'lstm_pred': 'mean',
                'final_pred': 'mean'
            }).reset_index()
            df_out['ds'] = df_out['date'].astype(str)
            df_out = df_out.drop(columns=['date'])

        # Fix JSON compliance issue with NaN values (stronger conversion)
        df_out = df_out.astype(object).where(pd.notnull(df_out), None)

        return df_out

    except Exception as e:
        print(f"Error in predict_weather: {e}")
        return None


# ============== ROUTES ==============

@app.get("/health")
def health():
    return {"status": "healthy", "service": "forecast_api"}


@app.get("/")
def root():
    return {
        "message": "Weather ML API (Prophet + LSTM) đang chạy",
        "model": "Hybrid Ensemble: 0.6*Prophet + 0.4*LSTM",
        "architecture": "Pure Orchestrator — calls prophet_api + lstm_api via HTTP",
        "supported_cities": list(CITIES.keys())
    }


@app.post("/predict")
def predict(request: ForecastRequest):
    try:
        result = predict_weather(
            hours=request.hours,
            days=request.days,
            mode=request.mode,
            city=request.city
        )

        if result is None or len(result) == 0:
            return {"status": "error", "message": "Prediction failed"}

        city_info = get_city_coords(request.city)

        return {
            "status": "success",
            "city": request.city,
            "city_name": city_info.get("name", request.city),
            "mode": request.mode,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data": result.to_dict(orient="records")
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
