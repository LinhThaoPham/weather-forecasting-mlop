"""MLOps Retrain Pipeline — Automated model retraining.

Pipeline Steps:
  1. Fetch data + Train Prophet (hourly + daily)
  2. Train LSTM (hourly + daily)
  3. Evaluate: compare new vs old models (MAE, RMSE)
  4. Version: archive old models, deploy new to current/
  5. Reload: call /reload on prophet_api + lstm_api
  6. Log: update registry.json

Usage:
  python src/training/retrain_pipeline.py             # Run once
  python src/training/retrain_pipeline.py --daemon     # Run as scheduler
"""
import os
import sys
import json
import shutil
import argparse
from datetime import datetime, timedelta

import requests
import numpy as np
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config.cities import CITIES
from src.config.constants import (
    BATCH_SIZE,
    DAILY_LOOKBACK,
    DAILY_HORIZON,
    DEFAULT_CITY,
    HISTORICAL_DAYS_HOURLY,
    HOURLY_LOOKBACK,
    HOURLY_HORIZON,
    MAX_EPOCHS,
    NUM_CITIES,
    RETRAIN_INTERVAL_DAYS,
    TRAIN_SPLIT_RATIO,
    model_filename,
    scaler_filename,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CURRENT_DIR = os.path.join(MODELS_DIR, "current")
ARCHIVE_DIR = os.path.join(MODELS_DIR, "archive")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")

from src.config.settings import LSTM_API_URL, PROPHET_API_URL

CITY_IDS = sorted(CITIES.keys())

# Build MODEL_FILES dynamically for all cities
MODEL_FILES = []
for _cid in CITY_IDS:
    MODEL_FILES.append(model_filename("prophet", "hourly", _cid))
    MODEL_FILES.append(model_filename("prophet", "daily", _cid))
MODEL_FILES.extend([
    model_filename("lstm", "hourly"),
    model_filename("lstm", "daily"),
    scaler_filename("hourly"),
    scaler_filename("daily"),
])


# ===== Registry =====

def load_registry() -> dict:
    """Load model registry from disk."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current_version": None, "models": []}


def save_registry(registry: dict) -> None:
    """Save model registry to disk."""
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def get_current_metrics() -> dict | None:
    """Load metrics from current active version in registry."""
    registry = load_registry()
    for model in registry["models"]:
        if model["version"] == registry.get("current_version"):
            return model.get("metrics", {})
    return None


# ===== Training History =====

def _extract_history(keras_history) -> dict:
    """Extract Keras history into JSON-serializable dict."""
    return {
        key: [round(float(v), 6) for v in values]
        for key, values in keras_history.history.items()
    }


def save_training_history(version_tag: str, history_data: dict) -> None:
    """Append training history (epoch-level curves) for monitoring."""
    entries = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, ValueError):
            entries = []

    entries.append({
        "version": version_tag,
        "trained_at": datetime.now().isoformat(),
        **history_data,
    })

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"📈 Training history saved ({len(entries)} versions)")


# ===== Versioning =====

def archive_current(version_tag: str) -> None:
    """Archive current models to archive/version_tag/."""
    archive_path = os.path.join(ARCHIVE_DIR, version_tag)
    os.makedirs(archive_path, exist_ok=True)

    for filename in MODEL_FILES:
        src = os.path.join(CURRENT_DIR, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(archive_path, filename))

    print(f"📦 Archived current models to {archive_path}")


def deploy_to_current(source_dir: str) -> None:
    """Copy trained models from source_dir to current/."""
    os.makedirs(CURRENT_DIR, exist_ok=True)
    for filename in MODEL_FILES:
        src = os.path.join(source_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(CURRENT_DIR, filename))
    print(f"🟢 Deployed models to {CURRENT_DIR}")


def reload_services() -> dict:
    """Call /reload on Prophet and LSTM services for hot model swap."""
    results = {}
    for name, url in [("prophet_api", PROPHET_API_URL), ("lstm_api", LSTM_API_URL)]:
        try:
            resp = requests.post(f"{url}/reload", timeout=30)
            results[name] = resp.json()
            print(f"♻️ {name} reloaded: {results[name]}")
        except requests.RequestException as e:
            print(f"⚠ Failed to reload {name}: {e}")
            results[name] = {"status": "error", "message": str(e)}
    return results


def fetch_training_data(city: str = DEFAULT_CITY):
    """Fetch training data FROM SQLite for a single city.

    Returns DataFrame with columns: ds, y, humidity, cloud_cover
    """
    from src.config.db import get_connection

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT timestamp, temperature, humidity, cloud_cover
               FROM weather_historical
               WHERE city_id = ?
               ORDER BY timestamp""",
            (city,),
        ).fetchall()

    if not rows:
        raise ValueError(f"No historical data in DB for city={city}. Run seed_database.py first.")

    import pandas as pd
    df = pd.DataFrame(rows, columns=["ds", "y", "humidity", "cloud_cover"])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna(subset=["y"])
    return df


def train_prophet_models(train_dir: str) -> dict:
    """Step 1: Train Prophet hourly + daily × 6 cities, return metrics."""
    from src.data_pipeline.feature_engineering import add_features
    from src.models_logic.prophet_model import train_prophet, save_prophet
    from src.training.evaluate import evaluate_prophet

    metrics = {}

    for city_id in CITY_IDS:
        print(f"\n   --- Prophet: {city_id} ---")
        weather_data = fetch_training_data(city_id)
        print(f"   📡 {city_id}: {len(weather_data):,} rows")

        # Hourly
        hourly_data = add_features(weather_data.copy(), is_hourly=True)
        split_idx = int(len(hourly_data) * TRAIN_SPLIT_RATIO)
        hourly_train = hourly_data.iloc[:split_idx]
        hourly_test = hourly_data.iloc[split_idx:]

        prophet_hourly = train_prophet(hourly_train, is_hourly=True)
        save_prophet(prophet_hourly, os.path.join(train_dir, model_filename("prophet", "hourly", city_id)))

        h_metrics = evaluate_prophet(prophet_hourly, hourly_test, mode="hourly")
        metrics[f"prophet_hourly_{city_id}"] = h_metrics
        print(f"   ✓ Hourly — MAE: {h_metrics['mae']}°C")

        # Daily
        daily_data = add_features(weather_data.copy(), is_hourly=False)
        split_idx = int(len(daily_data) * TRAIN_SPLIT_RATIO)
        daily_train = daily_data.iloc[:split_idx]
        daily_test = daily_data.iloc[split_idx:]

        prophet_daily = train_prophet(daily_train, is_hourly=False)
        save_prophet(prophet_daily, os.path.join(train_dir, model_filename("prophet", "daily", city_id)))

        d_metrics = evaluate_prophet(prophet_daily, daily_test, mode="daily")
        metrics[f"prophet_daily_{city_id}"] = d_metrics
        print(f"   ✓ Daily  — MAE: {d_metrics['mae']}°C")

    return metrics


def train_lstm_models(weather_data, train_dir: str) -> dict:
    """Step 2: Train LSTM hourly + daily, both multi-city (6 cities)."""
    from src.data_pipeline.feature_engineering import (
        build_multi_city_daily,
        build_multi_city_hourly,
        normalize_multi_city,
        sliding_window,
    )
    from src.models_logic.lstm_model import create_lstm_model
    from src.training.evaluate import evaluate_lstm_multi_city

    metrics = {}
    history_curves = {}

    # ── Hourly Multi-City (6 cities × ~17,520 hours) ──
    print("   📡 Building multi-city hourly tensor from DB...")
    df_hourly, city_ids = build_multi_city_hourly()
    hourly_data = df_hourly.values  # (N_hours, 6)

    hourly_normalized, hourly_scaler = normalize_multi_city(hourly_data)
    hourly_inputs, hourly_targets = sliding_window(hourly_normalized, HOURLY_LOOKBACK, HOURLY_HORIZON)

    split = int(len(hourly_inputs) * TRAIN_SPLIT_RATIO)
    X_train, X_val = hourly_inputs[:split], hourly_inputs[split:]
    y_train, y_val = hourly_targets[:split], hourly_targets[split:]

    lstm_hourly = create_lstm_model(lookback_window=HOURLY_LOOKBACK, forecast_horizon=HOURLY_HORIZON, feature_dim=NUM_CITIES)
    h_history = lstm_hourly.train(X_train, y_train, X_val, y_val, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
    lstm_hourly.save(os.path.join(train_dir, model_filename("lstm", "hourly")))
    joblib.dump(hourly_scaler, os.path.join(train_dir, scaler_filename("hourly")))

    hourly_metrics = evaluate_lstm_multi_city(lstm_hourly.model, X_val, y_val, hourly_scaler)
    hourly_metrics["val_loss"] = round(float(h_history.history["val_loss"][-1]), 6)
    hourly_metrics["city_ids"] = city_ids
    metrics["lstm_hourly"] = hourly_metrics
    history_curves["lstm_hourly"] = _extract_history(h_history)
    print(f"   ✓ LSTM Hourly (Multi-City) — MAE: {hourly_metrics['mae']}°C, Val Loss: {hourly_metrics['val_loss']}")

    # ── Daily Multi-City (6 cities × ~730 days) ──
    print("   📡 Building multi-city daily tensor from DB...")
    df_daily, city_ids = build_multi_city_daily()
    daily_data = df_daily.values  # (N_days, 6)

    daily_normalized, daily_scaler = normalize_multi_city(daily_data)
    daily_inputs, daily_targets = sliding_window(daily_normalized, DAILY_LOOKBACK, DAILY_HORIZON)

    split = int(len(daily_inputs) * TRAIN_SPLIT_RATIO)
    X_train, X_val = daily_inputs[:split], daily_inputs[split:]
    y_train, y_val = daily_targets[:split], daily_targets[split:]

    lstm_daily = create_lstm_model(lookback_window=DAILY_LOOKBACK, forecast_horizon=DAILY_HORIZON, feature_dim=NUM_CITIES)
    d_history = lstm_daily.train(X_train, y_train, X_val, y_val, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
    lstm_daily.save(os.path.join(train_dir, model_filename("lstm", "daily")))
    joblib.dump(daily_scaler, os.path.join(train_dir, scaler_filename("daily")))

    daily_metrics = evaluate_lstm_multi_city(lstm_daily.model, X_val, y_val, daily_scaler)
    daily_metrics["val_loss"] = round(float(d_history.history["val_loss"][-1]), 6)
    daily_metrics["city_ids"] = city_ids
    metrics["lstm_daily"] = daily_metrics
    history_curves["lstm_daily"] = _extract_history(d_history)
    print(f"   ✓ LSTM Daily (Multi-City) — MAE: {daily_metrics['mae']}°C, Val Loss: {daily_metrics['val_loss']}")

    return metrics, history_curves


def evaluate_and_decide(all_metrics: dict) -> str:
    """Step 3: Compare new vs old models, return 'accept' or 'rollback'."""
    from src.training.evaluate import compare_models

    old_metrics = get_current_metrics()
    if not old_metrics:
        print("   No previous models found — accepting new models")
        return "accept"

    for model_name in ["prophet_hourly", "lstm_hourly"]:
        old_m = old_metrics.get(model_name)
        new_m = all_metrics.get(model_name)
        if old_m and new_m and compare_models(new_m, old_m) == "rollback":
            return "rollback"

    return "accept"


def version_and_deploy(train_dir: str, version_tag: str) -> None:
    """Step 4: Archive old models and deploy new ones."""
    has_current = os.path.exists(CURRENT_DIR) and any(
        os.path.exists(os.path.join(CURRENT_DIR, f)) for f in MODEL_FILES
    )
    if has_current:
        registry = load_registry()
        old_version = registry.get("current_version", "v_unknown")
        archive_current(old_version)

    deploy_to_current(train_dir)


def update_registry(version_tag: str, all_metrics: dict, decision: str) -> None:
    """Step 6: Update model registry."""
    registry = load_registry()
    registry["models"].append({
        "version": version_tag,
        "trained_at": datetime.now().isoformat(),
        "data_range": f"{(datetime.now() - timedelta(days=HISTORICAL_DAYS_HOURLY)).strftime('%Y-%m-%d')} -> {datetime.now().strftime('%Y-%m-%d')}",
        "metrics": all_metrics,
        "status": "active" if decision == "accept" else "rejected",
        "decision": decision,
    })
    if decision == "accept":
        registry["current_version"] = version_tag
    save_registry(registry)


# ===== Main Pipeline =====

def run_retrain_pipeline() -> bool:
    """Execute the full 6-step retraining pipeline."""
    print(f"\n{'=' * 60}")
    print(f"🔄 MLOps RETRAIN PIPELINE — {datetime.now().isoformat()}")
    print(f"{'=' * 60}\n")

    version_tag = f"v_{datetime.now().strftime('%Y-%m-%d')}"
    train_dir = os.path.join(MODELS_DIR, "training_temp")
    os.makedirs(train_dir, exist_ok=True)

    try:
        print("[1/6] Training Prophet × 6 cities...")
        prophet_metrics = train_prophet_models(train_dir)

        print("\n[2/6] Training LSTM (multi-city)...")
        lstm_metrics, lstm_history = train_lstm_models(None, train_dir)

        all_metrics = {**prophet_metrics, **lstm_metrics}

        print("\n[3/6] Evaluating (new vs old)...")
        decision = evaluate_and_decide(all_metrics)
        print(f"   📊 Decision: {decision.upper()}")

        print("\n[4/6] Versioning and deploying...")
        if decision == "accept":
            version_and_deploy(train_dir, version_tag)
        else:
            print("   ⏪ Rollback — keeping current models")

        print("\n[5/6] Reloading services...")
        if decision == "accept":
            reload_services()
        else:
            print("   Skipped (rollback)")

        print("\n[6/6] Updating registry + saving history...")
        update_registry(version_tag, all_metrics, decision)
        save_training_history(version_tag, lstm_history)

    except (ImportError, ValueError, OSError) as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    finally:
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)

    status = "✅ COMPLETE" if decision == "accept" else "⏪ ROLLBACK"
    print(f"\n{'=' * 60}")
    print(f"{status} — Version: {version_tag}")
    print(f"{'=' * 60}\n")

    return decision == "accept"


def run_daemon(interval_days: int = RETRAIN_INTERVAL_DAYS) -> None:
    """Run as daemon — retrain on schedule."""
    import schedule
    import time

    print(f"\n🔄 Retrain Scheduler Started (every {interval_days} days)")
    print(f"   Press Ctrl+C to stop\n")

    schedule.every(interval_days).days.do(run_retrain_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Retrain Pipeline")
    parser.add_argument("--daemon", action="store_true", help="Run as scheduler daemon")
    parser.add_argument("--interval", type=int, default=RETRAIN_INTERVAL_DAYS, help="Retrain interval in days")
    args = parser.parse_args()

    if args.daemon:
        run_daemon(interval_days=args.interval)
    else:
        run_retrain_pipeline()
