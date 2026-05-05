# services/forecast_api/main.py
"""
Forecast API — Pure ML Orchestrator.
Reads HISTORICAL data from DB → feeds to Prophet API + LSTM API → Ensemble.
Does NOT use any future/forecast data from external APIs.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
import numpy as np

from src.config.cities import CITIES, get_city_coords
from src.config.constants import LSTM_CITY_IDS
from src.config.settings import (
    DATA_API_URL,
    LSTM_API_URL,
    PROPHET_API_URL,
    SERVICE_CALL_TIMEOUT,
)
import requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✓ Forecast API started (Pure ML Orchestrator)")
    print(f"  DATA_API_URL:   {DATA_API_URL}")
    print(f"  PROPHET_API_URL: {PROPHET_API_URL}")
    print(f"  LSTM_API_URL:    {LSTM_API_URL}")
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
    days: int = 3
    hours: int = 72
    mode: str = "hourly"


# ============== DATA FROM DATA-API ==============

def get_historical_from_api(city: str, hours: int = 192) -> pd.DataFrame | None:
    """Lấy dữ liệu lịch sử từ data-api (tránh vấn đề DB ephemeral trên Cloud Run)."""
    try:
        days = max(hours // 24 + 1, 10)
        resp = requests.get(
            f"{DATA_API_URL}/historical",
            params={"city": city, "days": days},
            timeout=SERVICE_CALL_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()

        records = result.get("data", [])
        if not records:
            print(f"⚠ No historical data from data-api for {city}")
            return None

        df = pd.DataFrame(records)
        # data-api trả về cột 'time' và 'temperature_2m'
        time_col = "time" if "time" in df.columns else "timestamp"
        temp_col = "temperature_2m" if "temperature_2m" in df.columns else "temperature"
        df = df.rename(columns={time_col: "ds", temp_col: "y"})
        df = df[["ds", "y"]].dropna()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").tail(hours).reset_index(drop=True)
        return df

    except Exception as e:
        print(f"Error fetching historical from data-api for {city}: {e}")
        return None


def get_multi_city_history(hours: int = 192) -> dict[str, pd.DataFrame]:
    """Lấy dữ liệu lịch sử cho tất cả LSTM cities qua data-api."""
    city_data = {}
    for city_id in sorted(LSTM_CITY_IDS):
        df = get_historical_from_api(city_id, hours=hours)
        if df is not None and len(df) > 0:
            city_data[city_id] = df
    return city_data


# ============== SERVICE CALLS ==============

def call_prophet_api(future_ds: list[str], mode: str = "hourly", city: str = "hanoi"):
    """Call Prophet API with timestamps + city. Returns dict with predictions + confidence."""
    try:
        data = [{"ds": ds} for ds in future_ds]

        response = requests.post(
            f"{PROPHET_API_URL}/forecast",
            json={"data": data, "mode": mode, "city": city},
            timeout=SERVICE_CALL_TIMEOUT,
        )
        result = response.json()

        if result.get("status") == "success":
            return {
                "predictions": np.array(result["predictions"]),
                "confidence_lower": result.get("confidence_lower"),
                "confidence_upper": result.get("confidence_upper"),
            }
        else:
            print(f"⚠ Prophet API error: {result}")
            return None

    except Exception as e:
        print(f"⚠ Prophet API call failed: {e}")
        return None


def call_prophet_multi_var(future_ds: list[str], city: str = "hanoi"):
    """Call Prophet API to get humidity, wind_speed, cloud_cover predictions."""
    try:
        data = [{"ds": ds} for ds in future_ds]
        response = requests.post(
            f"{PROPHET_API_URL}/forecast_multi_var",
            json={"data": data, "city": city},
            timeout=SERVICE_CALL_TIMEOUT,
        )
        result = response.json()
        if result.get("status") == "success":
            return result.get("predictions", {})
        else:
            print(f"⚠ Prophet multi-var error: {result}")
            return {}
    except Exception as e:
        print(f"⚠ Prophet multi-var call failed: {e}")
        return {}


def call_lstm_api_multi_city(city_temps: dict[str, list[float]], mode: str = "hourly"):
    """Call LSTM API with multi-city temperature data.

    LSTM was trained with 3 cities (hanoi, danang, hcm) as features.
    We must send all 3 to match the training shape.

    Args:
        city_temps: {"danang": [t1, t2, ...], "hanoi": [...], "hcm": [...]}
        mode: "hourly" or "daily"
    """
    try:
        # Build ordered sequences matching training order (alphabetical)
        ordered_cities = sorted(LSTM_CITY_IDS)
        sequences = []
        for city_id in ordered_cities:
            temps = city_temps.get(city_id, [])
            sequences.append(temps)

        response = requests.post(
            f"{LSTM_API_URL}/forecast_multi",
            json={"city_temperatures": sequences, "city_ids": ordered_cities, "mode": mode},
            timeout=SERVICE_CALL_TIMEOUT,
        )
        result = response.json()

        if result.get("status") == "success":
            return result["predictions"]  # dict: {city_id: [preds]}
        else:
            print(f"⚠ LSTM API error: {result}")
            return None

    except Exception as e:
        print(f"⚠ LSTM API call failed: {e}")
        return None


# ============== ENSEMBLE ==============

def get_dynamic_weights() -> tuple[float, float]:
    """Calculate ensemble weights based on recent model performance.

    Reads performance_history.json to determine which model
    has been more accurate recently, and assigns higher weight accordingly.
    Falls back to 0.5/0.5 if no history available.
    """
    import json
    import os

    history_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "registry.json"
    )
    history_path = os.path.normpath(history_path)

    try:
        if not os.path.exists(history_path):
            return 0.5, 0.5

        with open(history_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        models = registry.get("models", [])
        if not models:
            return 0.5, 0.5

        # Get latest accepted model metrics
        latest = None
        for m in reversed(models):
            if m.get("decision") == "accept":
                latest = m
                break

        if latest is None:
            return 0.5, 0.5

        metrics = latest.get("metrics", {})
        prophet_mae = None
        lstm_mae = None

        # Average Prophet MAE across cities
        prophet_maes = []
        for key, val in metrics.items():
            if key.startswith("prophet_hourly") and isinstance(val, dict):
                mae = val.get("mae")
                if mae is not None and mae != float("inf") and not np.isnan(mae):
                    prophet_maes.append(mae)
        if prophet_maes:
            prophet_mae = np.mean(prophet_maes)

        # LSTM MAE
        lstm_m = metrics.get("lstm_hourly", {})
        if isinstance(lstm_m, dict):
            mae = lstm_m.get("mae")
            if mae is not None and mae != float("inf") and not np.isnan(mae):
                lstm_mae = mae

        if prophet_mae is None or lstm_mae is None:
            return 0.5, 0.5

        # Inverse MAE weighting: lower MAE → higher weight
        # w_prophet = (1/prophet_mae) / (1/prophet_mae + 1/lstm_mae)
        if prophet_mae == 0:
            return 0.9, 0.1
        if lstm_mae == 0:
            return 0.1, 0.9

        inv_prophet = 1.0 / prophet_mae
        inv_lstm = 1.0 / lstm_mae
        total = inv_prophet + inv_lstm

        prophet_w = round(inv_prophet / total, 2)
        lstm_w = round(1.0 - prophet_w, 2)

        # Clamp weights to [0.3, 0.7] to prevent one model dominating completely
        prophet_w = max(0.3, min(0.7, prophet_w))
        lstm_w = 1.0 - prophet_w

        return prophet_w, lstm_w

    except Exception as e:
        print(f"⚠ Dynamic weight calculation failed ({e}), using 0.5/0.5")
        return 0.5, 0.5


def ensemble_predictions(prophet_pred, lstm_pred, prophet_weight=None, lstm_weight=None):
    """Blend Prophet and LSTM predictions with dynamic weights.

    If weights not provided, calculates them from recent model performance.
    """
    if lstm_pred is None:
        return prophet_pred
    if prophet_pred is None:
        return lstm_pred

    if prophet_weight is None or lstm_weight is None:
        prophet_weight, lstm_weight = get_dynamic_weights()

    print(f"  ⚖ Ensemble weights: Prophet={prophet_weight:.0%}, LSTM={lstm_weight:.0%}")

    min_len = min(len(prophet_pred), len(lstm_pred))
    return prophet_weight * np.array(prophet_pred[:min_len]) + lstm_weight * np.array(lstm_pred[:min_len])


# ============== MAIN PREDICTION ==============

def predict_weather(hours=72, days=3, mode="hourly", city="hanoi"):
    """Main prediction pipeline — PURE ML, no future data.

    1. Read historical data from DB (all 3 cities for LSTM)
    2. LSTM: feed 3-city history → predict future
    3. Prophet: feed timestamps → predict based on learned seasonality
    4. Ensemble → return results
    """
    try:
        # ── Step 1: Get multi-city history for LSTM ──
        city_data = get_multi_city_history(hours=192)
        if len(city_data) == 0:
            print("❌ No historical data for any city")
            return None

        # Check target city has data
        if city not in city_data and city in LSTM_CITY_IDS:
            print(f"❌ No historical data for target city {city}")
            return None

        now = pd.Timestamp.now().floor('h')

        # ── Step 2: Prepare multi-city input for LSTM ──
        if mode == "daily":
            city_temps = {}
            for cid, df in city_data.items():
                df_daily = df.copy()
                df_daily['date'] = df_daily['ds'].dt.date
                df_daily = df_daily.groupby('date').agg({'y': 'mean'}).reset_index()
                city_temps[cid] = df_daily['y'].values[-90:].tolist()
        else:
            city_temps = {}
            for cid, df in city_data.items():
                city_temps[cid] = df['y'].values[-24:].tolist()

        # Call LSTM API (multi-city)
        lstm_all_preds = call_lstm_api_multi_city(city_temps, mode=mode)
        lstm_preds = None
        if lstm_all_preds and city in lstm_all_preds:
            lstm_preds = np.array(lstm_all_preds[city])

        # ── Step 3: Generate future timestamps for Prophet ──
        if mode == "daily":
            future_timestamps = pd.date_range(
                start=now + pd.Timedelta(days=1),
                periods=days,
                freq='D'
            )
        else:
            future_timestamps = pd.date_range(
                start=now + pd.Timedelta(hours=1),
                periods=hours,
                freq='h'
            )

        future_ds = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_timestamps]
        prophet_result = call_prophet_api(future_ds, mode=mode, city=city)

        # Unpack prophet result (now returns dict with predictions + confidence)
        prophet_preds = None
        confidence_lower = None
        confidence_upper = None
        if prophet_result is not None:
            prophet_preds = prophet_result["predictions"]
            confidence_lower = prophet_result.get("confidence_lower")
            confidence_upper = prophet_result.get("confidence_upper")

        # ── Step 4: Ensemble ──
        if prophet_preds is None and lstm_preds is None:
            print("❌ Both Prophet and LSTM failed")
            return None

        final_preds = ensemble_predictions(
            prophet_preds.tolist() if prophet_preds is not None else None,
            lstm_preds.tolist() if lstm_preds is not None else None,
        )

        # ── Step 5: Format output ──
        out_len = len(final_preds) if final_preds is not None else 0
        if out_len == 0:
            if prophet_preds is not None:
                final_preds = prophet_preds
                out_len = len(prophet_preds)
            elif lstm_preds is not None:
                final_preds = lstm_preds
                out_len = len(lstm_preds)
            else:
                return None

        df_out = pd.DataFrame({
            'ds': future_timestamps[:out_len].astype(str),
            'prophet_pred': prophet_preds[:out_len].tolist() if prophet_preds is not None else [None] * out_len,
            'lstm_pred': lstm_preds[:out_len].tolist() if lstm_preds is not None else [None] * out_len,
            'final_pred': np.array(final_preds[:out_len]).tolist(),
        })

        # Confidence intervals from Prophet
        if confidence_lower and len(confidence_lower) >= out_len:
            df_out['confidence_lower'] = [round(v, 1) for v in confidence_lower[:out_len]]
        if confidence_upper and len(confidence_upper) >= out_len:
            df_out['confidence_upper'] = [round(v, 1) for v in confidence_upper[:out_len]]

        # ── Step 6: Get extra variables from Prophet multi-var ──
        extra_vars = call_prophet_multi_var(future_ds, city=city)
        if extra_vars:
            for var_name, var_preds in extra_vars.items():
                if var_preds and len(var_preds) >= out_len:
                    df_out[var_name] = [round(v, 1) for v in var_preds[:out_len]]
                elif var_preds:
                    padded = var_preds + [var_preds[-1]] * (out_len - len(var_preds))
                    df_out[var_name] = [round(v, 1) for v in padded[:out_len]]

        df_out = df_out.astype(object).where(pd.notnull(df_out), None)
        return df_out

    except Exception as e:
        print(f"Error in predict_weather: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============== ROUTES ==============

@app.get("/health")
def health():
    return {"status": "healthy", "service": "forecast_api"}


@app.get("/")
def root():
    return {
        "message": "Weather ML API (Prophet + LSTM)",
        "model": "Hybrid Ensemble: Dynamic weights (Prophet + LSTM based on recent MAE)",
        "architecture": "Pure ML — reads history from DB, models predict future",
        "data_source": "SQLite (weather_historical) — NO external forecast API used",
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
            return {"status": "error", "message": "Prediction failed — check DB has historical data"}

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
