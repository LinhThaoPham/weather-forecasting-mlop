"""Drift Detection — Compare AI predictions vs actual data.

Core MLOps concept: Only retrain when model performance degrades.
Compares yesterday's AI predictions against today's actual observations.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config.cities import CITIES
from src.config.db import get_connection
from src.config.gcp import USE_BIGQUERY

# Drift threshold in °C — retrain only if MAE exceeds this
DRIFT_THRESHOLD = 2.0


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


def compute_drift(city_id: str) -> dict:
    """Compare yesterday's AI predictions with actual observations.

    Returns:
        dict with keys: mae, n_compared, drifted, message
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    preds = get_yesterday_predictions(city_id)
    if preds is None or len(preds) == 0:
        return {
            "mae": None,
            "n_compared": 0,
            "drifted": None,
            "message": f"No AI predictions found for {city_id} on {yesterday}",
        }

    actuals = get_actual_observations(city_id, yesterday)
    if actuals is None or len(actuals) == 0:
        return {
            "mae": None,
            "n_compared": 0,
            "drifted": None,
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
            "drifted": None,
            "message": f"No matching timestamps for {city_id} on {yesterday}",
        }

    mae = float(np.mean(np.abs(merged["predicted_temp"] - merged["temperature"])))
    drifted = mae > DRIFT_THRESHOLD

    return {
        "mae": round(mae, 2),
        "n_compared": len(merged),
        "drifted": drifted,
        "message": (
            f"{city_id}: MAE={mae:.2f}°C ({'DRIFT DETECTED' if drifted else 'OK'}) "
            f"[{len(merged)} hours compared]"
        ),
    }


def check_drift_all_cities() -> dict:
    """Run drift detection across all cities.

    Returns:
        dict with:
            - should_retrain: bool
            - city_results: per-city drift info
            - summary: human-readable summary
    """
    results = {}
    drift_cities = []

    for city_id in CITIES:
        result = compute_drift(city_id)
        results[city_id] = result
        print(f"  📊 {result['message']}")

        if result["drifted"] is True:
            drift_cities.append(city_id)

    # Retrain if ANY city has drifted
    should_retrain = len(drift_cities) > 0

    # Edge case: no predictions existed (first run) → retrain anyway
    has_any_comparison = any(r["n_compared"] > 0 for r in results.values())
    if not has_any_comparison:
        print("  ⚠ No previous predictions found — first run, will retrain")
        should_retrain = True

    summary = (
        f"Drift check: {len(drift_cities)}/{len(CITIES)} cities drifted "
        f"(threshold: {DRIFT_THRESHOLD}°C) → "
        f"{'RETRAIN' if should_retrain else 'SKIP'}"
    )
    print(f"\n  🎯 {summary}")

    return {
        "should_retrain": should_retrain,
        "city_results": results,
        "summary": summary,
        "threshold": DRIFT_THRESHOLD,
    }


def save_ai_predictions(city_id: str, predictions: list[dict], model_version: str) -> int:
    """Save AI predictions to DB for tomorrow's drift comparison.

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
