"""Daily Weather Pipeline — Fetch data + Drift check + Retrain if needed.

Designed to run daily via:
  - GitHub Actions (cron schedule)
  - Windows Task Scheduler
  - Manual: python daily_pipeline.py

Pipeline flow:
  1. Fetch yesterday's actual weather (historical + current)
  2. Drift Detection: compare AI predictions vs actual → retrain?
  3. (Conditional) Retrain models if drift detected
  4. Run inference → save AI predictions for tomorrow's drift check
"""
import os
import sys
import time
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def step_fetch_today():
    """Step 1: Fetch yesterday's actual weather data for all cities."""
    from src.data_pipeline.fetch_data import run_daily_fetch
    run_daily_fetch(days=2)


def step_db_stats():
    """Step 2: Print DB statistics."""
    from src.config.db import get_connection

    with get_connection() as conn:
        hist = conn.execute("SELECT COUNT(*) FROM weather_historical").fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM weather_current").fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(timestamp) FROM weather_historical"
        ).fetchone()[0]

        # Check AI predictions table
        try:
            ai_preds = conn.execute("SELECT COUNT(*) FROM weather_ai_predictions").fetchone()[0]
        except Exception:
            ai_preds = 0

    print(f"\n--- DB Stats ---")
    print(f"   Historical:     {hist:,} rows (latest: {latest})")
    print(f"   Current:        {cur:,} rows")
    print(f"   AI Predictions: {ai_preds:,} rows")


def step_drift_check() -> bool:
    """Step 3: Compare yesterday's AI predictions vs actual data.

    Returns:
        True if retraining is needed, False otherwise.
    """
    from src.monitoring.drift_detector import check_drift_all_cities

    print("\n--- Drift Detection ---")
    result = check_drift_all_cities()
    return result["should_retrain"]


def step_retrain():
    """Step 4: Retrain Prophet + LSTM models."""
    from src.training.retrain_pipeline import run_retrain_pipeline

    print("\n--- Model Retrain ---")
    result = run_retrain_pipeline()
    print(f"\nRetrain result: {'SUCCESS' if result else 'FAILED'}")
    return result


def step_save_predictions():
    """Step 5: Run inference for all cities and save predictions to DB.

    These predictions will be compared against actual data TOMORROW
    to decide if retraining is needed (drift detection).
    """
    from src.config.cities import CITIES
    from src.config.db import get_connection, init_db
    from src.monitoring.drift_detector import save_ai_predictions
    import pandas as pd
    import numpy as np

    init_db()

    print("\n--- Saving AI Predictions ---")

    version_tag = f"v_{datetime.now().strftime('%Y-%m-%d')}"
    total_saved = 0

    for city_id in CITIES:
        try:
            # Read historical data from DB
            with get_connection() as conn:
                rows = conn.execute(
                    """SELECT timestamp as ds, temperature as y
                       FROM weather_historical
                       WHERE city_id = ?
                       ORDER BY timestamp DESC
                       LIMIT 192""",
                    (city_id,),
                ).fetchall()

            if not rows:
                print(f"  ⚠ No historical data for {city_id}, skipping")
                continue

            df_hist = pd.DataFrame([dict(r) for r in rows])
            df_hist['ds'] = pd.to_datetime(df_hist['ds'])
            df_hist = df_hist.sort_values('ds').reset_index(drop=True)

            # Get last 24 hours for LSTM input
            hist_temps = df_hist['y'].values[-24:]

            # Generate future timestamps
            now = pd.Timestamp.now().floor('h')
            future_times = pd.date_range(start=now + pd.Timedelta(hours=1), periods=72, freq='h')

            # Call LSTM API for predictions
            try:
                import requests
                from src.config.settings import LSTM_API_URL

                resp = requests.post(
                    f"{LSTM_API_URL}/forecast",
                    json={"temperatures": hist_temps.tolist(), "mode": "hourly"},
                    timeout=30,
                )
                result = resp.json()
                if result.get("status") == "success":
                    lstm_preds = np.array(result["predictions"])
                else:
                    lstm_preds = None
            except Exception as e:
                print(f"  ⚠ LSTM API unavailable for {city_id}: {e}")
                lstm_preds = None

            # Call Prophet API for predictions
            try:
                from src.config.settings import PROPHET_API_URL

                future_ds = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_times]
                data = [{"ds": ds} for ds in future_ds]

                resp = requests.post(
                    f"{PROPHET_API_URL}/forecast",
                    json={"data": data, "mode": "hourly"},
                    timeout=30,
                )
                result = resp.json()
                if result.get("status") == "success":
                    prophet_preds = np.array(result["predictions"])
                else:
                    prophet_preds = None
            except Exception as e:
                print(f"  ⚠ Prophet API unavailable for {city_id}: {e}")
                prophet_preds = None

            # Get extra variable predictions (humidity, wind, cloud)
            extra_vars = {}
            try:
                from src.config.settings import PROPHET_API_URL
                resp_extra = requests.post(
                    f"{PROPHET_API_URL}/forecast_multi_var",
                    json={"data": data, "city": city_id},
                    timeout=30,
                )
                extra_result = resp_extra.json()
                if extra_result.get("status") == "success":
                    extra_vars = extra_result.get("predictions", {})
            except Exception as e:
                print(f"  ⚠ Prophet multi-var unavailable for {city_id}: {e}")

            # Ensemble
            if prophet_preds is not None and lstm_preds is not None:
                min_len = min(len(prophet_preds), len(lstm_preds))
                final_preds = 0.6 * prophet_preds[:min_len] + 0.4 * lstm_preds[:min_len]
            elif prophet_preds is not None:
                final_preds = prophet_preds
            elif lstm_preds is not None:
                final_preds = lstm_preds
            else:
                print(f"  ⚠ No model available for {city_id}")
                continue

            # Save to DB (with multi-variable predictions)
            n_preds = min(len(final_preds), len(future_times))
            humidity_preds = extra_vars.get("humidity")
            wind_preds = extra_vars.get("wind_speed")
            cloud_preds = extra_vars.get("cloud_cover")

            predictions = [
                {
                    "target_time": future_times[i].strftime("%Y-%m-%d %H:%M:%S"),
                    "predicted_temp": round(float(final_preds[i]), 1),
                    "predicted_humidity": round(float(humidity_preds[i]), 1) if humidity_preds and i < len(humidity_preds) else None,
                    "predicted_wind_speed": round(float(wind_preds[i]), 1) if wind_preds and i < len(wind_preds) else None,
                    "predicted_cloud_cover": round(float(cloud_preds[i]), 1) if cloud_preds and i < len(cloud_preds) else None,
                }
                for i in range(n_preds)
            ]

            saved = save_ai_predictions(city_id, predictions, version_tag)
            total_saved += saved
            print(f"  ✓ {city_id}: {saved} predictions saved (temp + humidity + wind + cloud)")

        except Exception as e:
            print(f"  ⚠ Failed for {city_id}: {e}")

    print(f"\n  Total: {total_saved} predictions saved for drift check tomorrow")


def step_sync_gcs():
    """Step 6: Sync DB + models to GCS for Cloud Run deployment."""
    from src.config.gcp import USE_GCS
    from src.config.gcs_storage import upload_file, upload_models_dir
    from src.config.settings import DB_PATH

    if not USE_GCS:
        print("  ⚠ GCS disabled (USE_GCS=false) — skipping cloud sync")
        return

    print("\n--- GCS Sync ---")
    try:
        # Upload DB
        upload_file(DB_PATH, "data/weather_forecast.db")
        print("  ✓ DB uploaded to GCS")

        # Upload models
        import os
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "current")
        uploaded = upload_models_dir(models_dir)
        print(f"  ✓ {uploaded} model artifacts uploaded to GCS")

    except Exception as e:
        print(f"  ⚠ GCS sync failed: {e}")


def main():
    start = time.time()
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{'='*60}")
    print(f"Daily Weather Pipeline — {today}")
    print(f"{'='*60}")

    # Step 1: Fetch actual weather data
    print("\n[STEP 1] Fetching actual weather data...")
    step_fetch_today()

    # Step 2: DB Stats
    step_db_stats()

    # Step 3: Drift Detection
    print("\n[STEP 3] Checking model drift...")
    should_retrain = step_drift_check()

    # Step 4: Conditional Retrain
    if should_retrain:
        print("\n[STEP 4] Drift detected — Retraining models...")
        try:
            step_retrain()
        except Exception as e:
            print(f"⚠ Retrain failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[STEP 4] No drift — Skipping retrain (model still accurate)")

    # Step 5: Save predictions for tomorrow's drift check
    print("\n[STEP 5] Running inference and saving predictions...")
    try:
        step_save_predictions()
    except Exception as e:
        print(f"⚠ Save predictions failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Sync to GCS (for Cloud Run)
    print("\n[STEP 6] Syncing to GCS...")
    step_sync_gcs()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

