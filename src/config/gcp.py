"""GCP settings shared across storage, BigQuery, Cloud Run, and monitoring."""
import os


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "weather-forecasting-494811")
GCP_REGION = os.getenv("GCP_REGION", "asia-southeast1")

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"{GCP_PROJECT_ID}-models")
GCS_MODELS_PREFIX = os.getenv("GCS_MODELS_PREFIX", "models/current")

BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "weather_mlops")
BIGQUERY_HISTORICAL_TABLE = os.getenv("BIGQUERY_HISTORICAL_TABLE", "weather_historical")

USE_GCS = _as_bool(os.getenv("USE_GCS"), default=False)
USE_BIGQUERY = _as_bool(os.getenv("USE_BIGQUERY"), default=False)
ENABLE_VERTEX_METRICS = _as_bool(os.getenv("ENABLE_VERTEX_METRICS"), default=False)

VERTEX_EXPERIMENT = os.getenv("VERTEX_EXPERIMENT", "weather-forecasting-retrain")
