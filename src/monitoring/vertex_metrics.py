"""Vertex AI experiment logging for retrain MAE metrics."""
from __future__ import annotations

from datetime import datetime
import re

from google.cloud import aiplatform

from src.config.gcp import ENABLE_VERTEX_METRICS, GCP_PROJECT_ID, GCP_REGION, VERTEX_EXPERIMENT


def log_retrain_metrics(version_tag: str, all_metrics: dict, decision: str) -> None:
    if not ENABLE_VERTEX_METRICS:
        return

    import math
    metric_payload = {}
    for key, value in all_metrics.items():
        if isinstance(value, dict) and "mae" in value:
            v = float(value["mae"])
            # Skip NaN / Infinity — Vertex AI rejects them
            if math.isfinite(v):
                metric_payload[f"{key}_mae"] = v

    if not metric_payload:
        return

    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION, experiment=VERTEX_EXPERIMENT)
    raw_name = f"retrain-{version_tag}-{datetime.now().strftime('%H%M%S')}".lower().replace("_", "-")
    run_name = re.sub(r"[^a-z0-9-]", "-", raw_name)
    run_name = re.sub(r"-{2,}", "-", run_name).strip("-")
    if not run_name:
        run_name = "retrain-run"
    run_name = run_name[:128]
    with aiplatform.start_run(run=run_name):
        aiplatform.log_params({"decision": decision, "version": version_tag})
        aiplatform.log_metrics(metric_payload)
