"""Daily fetch script: ingest latest weather rows and append BigQuery with dedup."""
from __future__ import annotations

from src.config.cities import CITIES
from src.config.db import init_db
from src.data_pipeline.bigquery_storage import ensure_dataset_and_table
from src.data_pipeline.store_data import store_current, store_forecast, store_historical


def run_daily_fetch(days: int = 2, forecast_days: int = 3) -> None:
    init_db()
    ensure_dataset_and_table()

    for city_id in CITIES:
        print(f"\n--- {city_id} ---")
        try:
            store_historical(city_id, days=days)
        except Exception as e:
            print(f"  ⚠ Historical fetch failed: {e}")

        try:
            store_forecast(city_id, days=forecast_days)
        except Exception as e:
            print(f"  ⚠ Forecast fetch failed: {e}")

        try:
            store_current(city_id)
        except Exception as e:
            print(f"  ⚠ Current fetch failed: {e}")


if __name__ == "__main__":
    run_daily_fetch()
