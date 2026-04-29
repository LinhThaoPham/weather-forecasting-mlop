"""Cloud Run Function — Data Ingester.

Triggered by Pub/Sub (Cloud Scheduler daily cron).
Fetches weather data for all 6 cities → stores in Cloud SQL.
"""
import os
import sys
import json
import functions_framework

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@functions_framework.cloud_event
def ingest_weather_data(cloud_event):
    """Pub/Sub trigger: Fetch today's weather for all cities."""
    from src.config.cities import CITIES
    from src.config.db import init_db, get_connection
    from src.data_pipeline.store_data import store_current, store_forecast, store_historical

    # Decode trigger message
    import base64
    if cloud_event.data and cloud_event.data.get("message", {}).get("data"):
        payload = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
        print(f"Trigger payload: {payload}")

    init_db()

    results = {"success": [], "failed": []}

    for city_id in CITIES:
        print(f"\n--- {city_id} ---")

        try:
            store_historical(city_id, days=2)
            print(f"  ✓ Historical OK")
        except Exception as e:
            print(f"  ⚠ Historical failed: {e}")
            results["failed"].append(f"{city_id}_historical")

        try:
            store_forecast(city_id, days=3)
            print(f"  ✓ Forecast OK")
        except Exception as e:
            print(f"  ⚠ Forecast failed: {e}")
            results["failed"].append(f"{city_id}_forecast")

        try:
            store_current(city_id)
            print(f"  ✓ Current OK")
        except Exception as e:
            print(f"  ⚠ Current failed: {e}")
            results["failed"].append(f"{city_id}_current")

        results["success"].append(city_id)

    # Log DB stats
    with get_connection() as conn:
        conn.execute("SELECT COUNT(*) FROM weather_historical")
        hist_count = conn.fetchone()[0]
        conn.execute("SELECT COUNT(*) FROM weather_forecast")
        fc_count = conn.fetchone()[0]

    print(f"\n📊 DB Stats: Historical={hist_count:,} | Forecast={fc_count:,}")

    # Trigger retrain if enough data
    if not results["failed"]:
        _trigger_retrain()

    return results


def _trigger_retrain():
    """Publish message to retrain trigger topic."""
    try:
        from google.cloud import pubsub_v1

        publisher = pubsub_v1.PublisherClient()
        project_id = os.getenv("GCP_PROJECT_ID", "")
        topic_path = publisher.topic_path(project_id, "model-retrain-trigger")

        data = json.dumps({"trigger": "retrain", "reason": "daily_ingest_complete"})
        publisher.publish(topic_path, data.encode("utf-8"))
        print("✅ Retrain trigger published")

    except Exception as e:
        print(f"⚠ Failed to trigger retrain: {e}")
