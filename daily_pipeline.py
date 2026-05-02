"""Daily Weather Pipeline — Fetch today's data + retrain models.

Designed to run daily via:
  - GitHub Actions (cron schedule)
  - Windows Task Scheduler
  - Manual: python daily_pipeline.py
"""
import os
import sys
import time
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def step_fetch_today():
    """Step 1: Fetch today's weather data for all 6 cities."""
    from src.data_pipeline.fetch_data import run_daily_fetch
    run_daily_fetch(days=2, forecast_days=3)


def step_db_stats():
    """Step 2: Print DB statistics."""
    from src.config.db import get_connection

    with get_connection() as conn:
        hist = conn.execute("SELECT COUNT(*) FROM weather_historical").fetchone()[0]
        fc = conn.execute("SELECT COUNT(*) FROM weather_forecast").fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM weather_current").fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(timestamp) FROM weather_historical"
        ).fetchone()[0]

    print(f"\n📊 DB Stats:")
    print(f"   Historical: {hist:,} rows (latest: {latest})")
    print(f"   Forecast:   {fc:,} rows")
    print(f"   Current:    {cur:,} rows")


def step_retrain():
    """Step 3: Retrain Prophet + LSTM models."""
    from src.training.retrain_pipeline import run_retrain_pipeline

    print("\n🔄 Starting model retrain...")
    result = run_retrain_pipeline()
    print(f"\n✅ Retrain result: {result}")
    return result


def main():
    start = time.time()
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{'='*60}")
    print(f"🌤  Daily Weather Pipeline — {today}")
    print(f"{'='*60}")

    # Step 1: Fetch
    print("\n📡 STEP 1: Fetching today's data...")
    step_fetch_today()

    # Step 2: Stats
    step_db_stats()

    # Step 3: Retrain
    print("\n🧠 STEP 3: Retraining models...")
    try:
        step_retrain()
    except Exception as e:
        print(f"⚠ Retrain failed: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
