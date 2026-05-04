#!/bin/bash
# setup_cloud_run_job.sh
# Chạy MỘT LẦN trên GCP VM hoặc Cloud Shell để tạo Cloud Run Job
# Usage: bash setup_cloud_run_job.sh

set -e

PROJECT_ID="weather-forecasting-494811"
REGION="asia-southeast1"
IMAGE="asia-southeast1-docker.pkg.dev/${PROJECT_ID}/weather-forecasting/weather-forecasting-mlops:latest"
JOB_NAME="retrain-job"

echo "=========================================="
echo "  Creating Cloud Run Job: ${JOB_NAME}"
echo "=========================================="

gcloud run jobs create "${JOB_NAME}" \
  --image "${IMAGE}" \
  --command "python" \
  --args "src/training/retrain_pipeline.py" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --memory 4Gi \
  --cpu 4 \
  --task-timeout 3600 \
  --max-retries 1 \
  --set-env-vars "USE_GCS=true,USE_BIGQUERY=false,ENABLE_VERTEX_METRICS=true,GCS_BUCKET_NAME=${PROJECT_ID}-models,GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}" \
  --service-account "github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

echo ""
echo "✅ Cloud Run Job '${JOB_NAME}' created!"
echo ""
echo "Test thủ công:"
echo "  gcloud run jobs execute ${JOB_NAME} --region ${REGION} --wait"
