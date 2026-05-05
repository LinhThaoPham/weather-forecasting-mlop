"""Model Performance Monitor — Track prediction accuracy over time.

Replaces the old "drift detection" which incorrectly triggered daily retrains.

Strategy:
  - Daily: compute prediction error (MAE) per city, save to performance_history.json
  - Retrain trigger: ONLY when MAE exceeds threshold for N consecutive days (sustained drift)
  - Cooldown: minimum 14 days between retrains to avoid thrashing
"""
import json
import os

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config.cities import CITIES
from src.config.db import get_connection
from src.config.gcp import USE_BIGQUERY
from src.config.constants import (
    DRIFT_THRESHOLD,
    RETRAIN_COOLDOWN_DAYS,
    SUSTAINED_DRIFT_DAYS,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PERFORMANCE_HISTORY_PATH = os.path.join(MODELS_DIR, "performance_history.json")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")


# ===== Data Fetching (unchanged) =====

def get_yesterday_predictions(city_id: str) -> pd.DataFrame | None:
    """Fetch AI predictions that targeted yesterday's timestamps."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if USE_BIGQUERY:
        from src.data_pipeline.bigquery_storage import fetch_yesterday_predictions
        df = fetch_yesterday_predictions(city_id, yesterday)
        if df.empty:
            return None
        df["target_time"] = pd.to_datetime(df["target_time"]).dt.tz_localize(None)
        return df

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT target_time, predicted_temp
               FROM weather_ai_predictions
               WHERE city_id = ?
                 AND target_time LIKE ?
               ORDER BY target_time""",
            (city_id, f"{yesterday}%"),
        ).fetchall()

    if not rows:
        return None

    df = pd.DataFrame([dict(r) for r in rows])
    df["target_time"] = pd.to_datetime(df["target_time"])
    return df


def get_actual_observations(city_id: str, date_str: str) -> pd.DataFrame | None:
    """Fetch actual historical observations for a specific date."""
    if USE_BIGQUERY:
        from src.data_pipeline.bigquery_storage import fetch_historical_df
        df = fetch_historical_df(city_id)
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        # Filter for the specific date
        df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == date_str]
        if df.empty:
            return None
        return df[["timestamp", "temperature"]]

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT timestamp, temperature
               FROM weather_historical
               WHERE city_id = ?
                 AND timestamp LIKE ?
               ORDER BY timestamp""",
            (city_id, f"{date_str}%"),
        ).fetchall()

    if not rows:
        return None

    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ===== Performance Computation =====

def compute_city_performance(city_id: str) -> dict:
    """Compare yesterday's AI predictions with actual observations for a city.

    Returns:
        dict with keys: mae, n_compared, exceeded_threshold, message
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    preds = get_yesterday_predictions(city_id)
    if preds is None or len(preds) == 0:
        return {
            "mae": None,
            "n_compared": 0,
            "exceeded_threshold": None,
            "message": f"No AI predictions found for {city_id} on {yesterday}",
        }

    actuals = get_actual_observations(city_id, yesterday)
    if actuals is None or len(actuals) == 0:
        return {
            "mae": None,
            "n_compared": 0,
            "exceeded_threshold": None,
            "message": f"No actual data found for {city_id} on {yesterday}",
        }

    # Merge on closest timestamp (hourly resolution)
    preds["hour"] = preds["target_time"].dt.floor("h")
    actuals["hour"] = actuals["timestamp"].dt.floor("h")

    merged = preds.merge(actuals, on="hour", how="inner")

    if len(merged) == 0:
        return {
            "mae": None,
            "n_compared": 0,
            "exceeded_threshold": None,
            "message": f"No matching timestamps for {city_id} on {yesterday}",
        }

    mae = float(np.mean(np.abs(merged["predicted_temp"] - merged["temperature"])))
    exceeded = mae > DRIFT_THRESHOLD

    return {
        "mae": round(mae, 2),
        "n_compared": len(merged),
        "exceeded_threshold": exceeded,
        "message": (
            f"{city_id}: MAE={mae:.2f}°C ({'⚠ EXCEEDED' if exceeded else '✓ OK'}) "
            f"[{len(merged)} hours compared]"
        ),
    }


# ===== Performance History =====

def load_performance_history() -> list:
    """Load daily performance history from disk."""
    if os.path.exists(PERFORMANCE_HISTORY_PATH):
        try:
            with open(PERFORMANCE_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def save_performance_history(entries: list) -> None:
    """Save performance history to disk."""
    with open(PERFORMANCE_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def check_daily_performance() -> dict:
    """Monitor model performance across all cities, save to history.

    This ONLY logs metrics — it does NOT trigger retraining.

    Returns:
        dict with avg_mae, city_results, exceeded_threshold
    """
    results = {}
    mae_values = []

    for city_id in CITIES:
        result = compute_city_performance(city_id)
        results[city_id] = result
        print(f"  📊 {result['message']}")

        if result["mae"] is not None:
            mae_values.append(result["mae"])

    avg_mae = round(float(np.mean(mae_values)), 2) if mae_values else None
    exceeded = avg_mae is not None and avg_mae > DRIFT_THRESHOLD

    today_entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "city_results": {
            cid: {"mae": r["mae"], "n_compared": r["n_compared"]}
            for cid, r in results.items()
        },
        "avg_mae": avg_mae,
        "exceeded_threshold": exceeded,
    }

    # Append to history
    history = load_performance_history()
    history.append(today_entry)
    # Keep last 90 days only
    history = history[-90:]
    save_performance_history(history)

    status = f"⚠ EXCEEDED ({avg_mae:.2f}°C > {DRIFT_THRESHOLD}°C)" if exceeded else f"✓ OK ({avg_mae:.2f}°C)" if avg_mae else "? No data"
    print(f"\n  🎯 Daily performance: {status}")
    print(f"  📁 History saved ({len(history)} entries)")

    return {
        "avg_mae": avg_mae,
        "city_results": results,
        "exceeded_threshold": exceeded,
    }


def check_sustained_drift() -> bool:
    """Check if performance has exceeded threshold for N consecutive days.

    Returns True ONLY if the last SUSTAINED_DRIFT_DAYS entries ALL exceeded
    the threshold. This prevents retraining on temporary bad days.
    """
    history = load_performance_history()

    if len(history) < SUSTAINED_DRIFT_DAYS:
        print(f"  📊 Not enough history ({len(history)}/{SUSTAINED_DRIFT_DAYS} days) — no sustained drift")
        return False

    recent = history[-SUSTAINED_DRIFT_DAYS:]
    all_exceeded = all(
        entry.get("exceeded_threshold", False) for entry in recent
    )

    if all_exceeded:
        dates = [e["date"] for e in recent]
        print(f"  🔴 Sustained drift detected: {SUSTAINED_DRIFT_DAYS} consecutive days ({dates[0]} → {dates[-1]})")
    else:
        exceeded_count = sum(1 for e in recent if e.get("exceeded_threshold", False))
        print(f"  ✓ No sustained drift: {exceeded_count}/{SUSTAINED_DRIFT_DAYS} recent days exceeded threshold")

    return all_exceeded


def check_cooldown() -> bool:
    """Check if enough time has passed since last retrain.

    Returns True if retrain is allowed (cooldown has elapsed).
    """
    if not os.path.exists(REGISTRY_PATH):
        return True

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return True

    models = registry.get("models", [])
    if not models:
        return True

    # Find last accepted retrain date
    last_retrain = None
    for model in reversed(models):
        if model.get("decision") == "accept":
            try:
                last_retrain = datetime.fromisoformat(model["trained_at"].split("+")[0])
            except (KeyError, ValueError):
                continue
            break

    if last_retrain is None:
        return True

    days_since = (datetime.now() - last_retrain).days
    allowed = days_since >= RETRAIN_COOLDOWN_DAYS

    if not allowed:
        print(f"  ⏳ Cooldown active: {days_since}/{RETRAIN_COOLDOWN_DAYS} days since last retrain")
    else:
        print(f"  ✓ Cooldown elapsed: {days_since} days since last retrain")

    return allowed


# ===== Legacy compat =====

def check_drift_all_cities() -> dict:
    """Legacy wrapper — now uses performance monitoring instead of instant retrain.

    Returns same structure for backward compatibility, but should_retrain
    is now based on sustained drift + cooldown, not single-day drift.
    """
    perf = check_daily_performance()
    sustained = check_sustained_drift()
    cooldown_ok = check_cooldown() if sustained else False

    should_retrain = sustained and cooldown_ok

    summary = (
        f"Performance check: avg MAE={perf['avg_mae']}°C "
        f"(threshold: {DRIFT_THRESHOLD}°C) → "
        f"{'RETRAIN (sustained drift + cooldown OK)' if should_retrain else 'SKIP'}"
    )
    print(f"\n  🎯 {summary}")

    return {
        "should_retrain": should_retrain,
        "city_results": perf["city_results"],
        "summary": summary,
        "threshold": DRIFT_THRESHOLD,
    }


def save_ai_predictions(city_id: str, predictions: list[dict], model_version: str) -> int:
    """Save AI predictions to DB for tomorrow's performance comparison.

    Args:
        city_id: city identifier
        predictions: list of dicts with target_time, predicted_temp,
                     and optionally: predicted_humidity, predicted_wind_speed, predicted_cloud_cover
        model_version: version tag (e.g. "v_2026-05-03")

    Returns:
        Number of rows inserted
    """
    if USE_BIGQUERY:
        from src.data_pipeline.bigquery_storage import append_ai_predictions
        return append_ai_predictions(city_id, predictions, model_version)

    with get_connection() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO weather_ai_predictions
               (city_id, target_time, predicted_temp, predicted_humidity,
                predicted_wind_speed, predicted_cloud_cover, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    city_id,
                    p["target_time"],
                    p.get("predicted_temp"),
                    p.get("predicted_humidity"),
                    p.get("predicted_wind_speed"),
                    p.get("predicted_cloud_cover"),
                    model_version,
                )
                for p in predictions
            ],
        )

    return len(predictions)
