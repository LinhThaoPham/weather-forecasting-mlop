"""BigQuery storage for historical weather rows with deduplication."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

import pandas as pd
from google.cloud import bigquery

from src.config.gcp import (
    BIGQUERY_DATASET,
    BIGQUERY_HISTORICAL_TABLE,
    GCP_PROJECT_ID,
    USE_BIGQUERY,
)


def _client() -> bigquery.Client:
    return bigquery.Client(project=GCP_PROJECT_ID)


def _table_ref() -> str:
    return f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_HISTORICAL_TABLE}"


def ensure_dataset_and_table() -> None:
    if not USE_BIGQUERY:
        return

    client = _client()
    dataset_ref = bigquery.Dataset(f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}")
    dataset_ref.location = "asia-southeast1"
    client.create_dataset(dataset_ref, exists_ok=True)

    sql = f"""
    CREATE TABLE IF NOT EXISTS `{_table_ref()}` (
      city_id STRING NOT NULL,
      timestamp TIMESTAMP NOT NULL,
      temperature FLOAT64,
      humidity FLOAT64,
      cloud_cover FLOAT64,
      apparent_temp FLOAT64,
      precipitation FLOAT64,
      rain FLOAT64,
      weather_code INT64,
      pressure FLOAT64,
      wind_speed FLOAT64,
      wind_direction FLOAT64,
      wind_gusts FLOAT64,
      dewpoint FLOAT64,
      fetched_at TIMESTAMP
    )
    PARTITION BY DATE(timestamp)
    CLUSTER BY city_id
    """
    client.query(sql).result()


def append_historical_rows(city_id: str, rows: list[dict[str, Any]]) -> int:
    """Append rows to BigQuery with key dedup on (city_id, timestamp)."""
    if not USE_BIGQUERY or not rows:
        return 0

    ensure_dataset_and_table()
    client = _client()
    now_utc = datetime.now(timezone.utc).isoformat()

    payload = []
    for r in rows:
        payload.append({
            "city_id": city_id,
            "timestamp": r.get("timestamp"),
            "temperature": r.get("temperature"),
            "humidity": r.get("humidity"),
            "cloud_cover": r.get("cloud_cover"),
            "apparent_temp": r.get("apparent_temp"),
            "precipitation": r.get("precipitation"),
            "rain": r.get("rain"),
            "weather_code": r.get("weather_code"),
            "pressure": r.get("pressure"),
            "wind_speed": r.get("wind_speed"),
            "wind_direction": r.get("wind_direction"),
            "wind_gusts": r.get("wind_gusts"),
            "dewpoint": r.get("dewpoint"),
            "fetched_at": now_utc,
        })

    temp_table = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}._hist_staging_{uuid.uuid4().hex[:12]}"
    schema = [
        bigquery.SchemaField("city_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("temperature", "FLOAT"),
        bigquery.SchemaField("humidity", "FLOAT"),
        bigquery.SchemaField("cloud_cover", "FLOAT"),
        bigquery.SchemaField("apparent_temp", "FLOAT"),
        bigquery.SchemaField("precipitation", "FLOAT"),
        bigquery.SchemaField("rain", "FLOAT"),
        bigquery.SchemaField("weather_code", "INT64"),
        bigquery.SchemaField("pressure", "FLOAT"),
        bigquery.SchemaField("wind_speed", "FLOAT"),
        bigquery.SchemaField("wind_direction", "FLOAT"),
        bigquery.SchemaField("wind_gusts", "FLOAT"),
        bigquery.SchemaField("dewpoint", "FLOAT"),
        bigquery.SchemaField("fetched_at", "TIMESTAMP"),
    ]

    load_job = client.load_table_from_json(
        payload,
        temp_table,
        job_config=bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE",
        ),
    )
    load_job.result()

    merge_sql = f"""
    MERGE `{_table_ref()}` AS target
    USING `{temp_table}` AS src
    ON target.city_id = src.city_id
       AND target.timestamp = src.timestamp
    WHEN NOT MATCHED THEN
      INSERT (
        city_id, timestamp, temperature, humidity, cloud_cover, apparent_temp,
        precipitation, rain, weather_code, pressure, wind_speed, wind_direction,
        wind_gusts, dewpoint, fetched_at
      )
      VALUES (
        src.city_id, src.timestamp, src.temperature, src.humidity, src.cloud_cover,
        src.apparent_temp, src.precipitation, src.rain, src.weather_code, src.pressure,
        src.wind_speed, src.wind_direction, src.wind_gusts, src.dewpoint, src.fetched_at
      )
    """
    client.query(merge_sql).result()
    client.delete_table(temp_table, not_found_ok=True)
    return len(payload)


def fetch_historical_df(city_id: str | None = None) -> pd.DataFrame:
    """Read historical rows from BigQuery into DataFrame.

    Returns only timestamp + temperature (Prophet is pure time-series).
    """
    if not USE_BIGQUERY:
        raise ValueError("USE_BIGQUERY is disabled")

    ensure_dataset_and_table()
    client = _client()

    where_clause = ""
    params: list[bigquery.ScalarQueryParameter] = []
    if city_id:
        where_clause = "WHERE city_id = @city_id"
        params.append(bigquery.ScalarQueryParameter("city_id", "STRING", city_id))

    sql = f"""
    SELECT city_id, timestamp, temperature
    FROM `{_table_ref()}`
    {where_clause}
    ORDER BY timestamp
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    )
    rows = list(job.result())
    if not rows:
        return pd.DataFrame(columns=["city_id", "timestamp", "temperature"])

    records = []
    for row in rows:
        records.append({
            "city_id": row["city_id"],
            "timestamp": row["timestamp"],
            "temperature": row["temperature"],
        })
    return pd.DataFrame(records)
