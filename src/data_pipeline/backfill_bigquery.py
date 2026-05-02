"""One-off historical backfill from Open-Meteo into SQLite + BigQuery."""
from __future__ import annotations

from src.config.db import init_db
from src.data_pipeline.bigquery_storage import ensure_dataset_and_table
from src.data_pipeline.store_data import seed_all_cities


def main() -> None:
    init_db()
    ensure_dataset_and_table()
    seed_all_cities()


if __name__ == "__main__":
    main()
