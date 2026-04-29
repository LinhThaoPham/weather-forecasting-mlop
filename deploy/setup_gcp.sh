#!/bin/bash
# ============================================================
# GCP Cloud Run — One-click Setup & Deploy Script
# ============================================================
# Usage:
#   chmod +x deploy/setup_gcp.sh
#   ./deploy/setup_gcp.sh <PROJECT_ID>
#
# Prerequisites:
#   - gcloud CLI installed & authenticated
#   - Billing enabled on GCP project
# ============================================================

set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy/setup_gcp.sh <PROJECT_ID>}"
REGION="asia-southeast1"
DB_INSTANCE="weather-db"
DB_NAME="weather"
DB_USER="weather_app"
GCS_BUCKET="weather-models-${PROJECT_ID}"

echo "============================================================"
echo "🚀 Setting up GCP project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "============================================================"

# ── Step 1: Set project & enable APIs ──
echo ""
echo "📦 [1/8] Enabling APIs..."
gcloud config set project "${PROJECT_ID}"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com \
    pubsub.googleapis.com \
    cloudscheduler.googleapis.com \
    cloudfunctions.googleapis.com \
    storage.googleapis.com

echo "✅ APIs enabled"

# ── Step 2: Create Artifact Registry ──
echo ""
echo "📦 [2/8] Creating Artifact Registry..."
gcloud artifacts repositories create weather \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Weather MLOps Docker images" \
    2>/dev/null || echo "   (already exists)"

echo "✅ Artifact Registry ready"

# ── Step 3: Create Cloud SQL ──
echo ""
echo "🐘 [3/8] Creating Cloud SQL (PostgreSQL 15)..."
gcloud sql instances create "${DB_INSTANCE}" \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region="${REGION}" \
    --storage-size=10GB \
    --storage-auto-increase \
    2>/dev/null || echo "   (already exists)"

# Create database
gcloud sql databases create "${DB_NAME}" \
    --instance="${DB_INSTANCE}" \
    2>/dev/null || echo "   (database already exists)"

# Generate random password
DB_PASS=$(openssl rand -base64 16)
gcloud sql users set-password "${DB_USER}" \
    --instance="${DB_INSTANCE}" \
    --password="${DB_PASS}" \
    2>/dev/null || \
gcloud sql users create "${DB_USER}" \
    --instance="${DB_INSTANCE}" \
    --password="${DB_PASS}"

INSTANCE_CONNECTION="${PROJECT_ID}:${REGION}:${DB_INSTANCE}"
echo "✅ Cloud SQL ready: ${INSTANCE_CONNECTION}"

# ── Step 4: Create GCS Bucket ──
echo ""
echo "🗂  [4/8] Creating GCS bucket..."
gsutil mb -l "${REGION}" "gs://${GCS_BUCKET}" 2>/dev/null || echo "   (already exists)"
echo "✅ GCS bucket: gs://${GCS_BUCKET}"

# Upload local models if they exist
if [ -d "models/current" ] && [ "$(ls models/current/ 2>/dev/null)" ]; then
    echo "   Uploading local models to GCS..."
    gsutil -m cp -r models/current/* "gs://${GCS_BUCKET}/current/"
    echo "   ✅ Models uploaded"
fi

# ── Step 5: Store Secrets ──
echo ""
echo "🔐 [5/8] Setting up Secret Manager..."

create_secret() {
    local name=$1
    local value=$2
    echo -n "${value}" | gcloud secrets create "${name}" --data-file=- 2>/dev/null || \
    echo -n "${value}" | gcloud secrets versions add "${name}" --data-file=-
}

create_secret "db-user" "${DB_USER}"
create_secret "db-password" "${DB_PASS}"
create_secret "db-name" "${DB_NAME}"
create_secret "db-instance" "${INSTANCE_CONNECTION}"

# Check if OWM key exists in .env
if [ -f ".env" ]; then
    OWM_KEY=$(grep OPENWEATHER_API_KEY .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    if [ -n "${OWM_KEY}" ]; then
        create_secret "owm-api-key" "${OWM_KEY}"
        echo "   ✅ OWM API key stored"
    fi
fi

echo "✅ Secrets configured"

# ── Step 6: Create Pub/Sub Topics ──
echo ""
echo "📬 [6/8] Creating Pub/Sub topics..."
gcloud pubsub topics create weather-data-ingest 2>/dev/null || echo "   (topic exists)"
gcloud pubsub topics create model-retrain-trigger 2>/dev/null || echo "   (topic exists)"
echo "✅ Pub/Sub topics ready"

# ── Step 7: Create Cloud Scheduler Job ──
echo ""
echo "⏰ [7/8] Creating Cloud Scheduler..."
gcloud scheduler jobs create pubsub daily-weather-ingest \
    --schedule="0 0 * * *" \
    --topic="weather-data-ingest" \
    --message-body='{"trigger": "daily_ingest"}' \
    --time-zone="Asia/Ho_Chi_Minh" \
    --location="${REGION}" \
    2>/dev/null || echo "   (scheduler already exists)"
echo "✅ Cloud Scheduler: daily at 07:00 VN"

# ── Step 8: Build & Deploy ──
echo ""
echo "🔨 [8/8] Building and deploying..."
gcloud builds submit --config deploy/cloudbuild.yaml \
    --substitutions="_REGION=${REGION}"

echo ""
echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "============================================================"
echo ""
echo "🔗 Service URLs:"
for svc in data-api forecast-api prophet-api lstm-api; do
    URL=$(gcloud run services describe "${svc}" --region="${REGION}" --format='value(status.url)' 2>/dev/null || echo "pending...")
    echo "   ${svc}: ${URL}"
done
echo ""
echo "📊 Dashboard: Deploy separately or use Firebase Hosting"
echo "🐘 Cloud SQL: ${INSTANCE_CONNECTION}"
echo "🗂  GCS: gs://${GCS_BUCKET}"
echo ""
echo "⚠️  Next steps:"
echo "   1. Initialize database: Run seed_database.py against Cloud SQL"
echo "   2. Update Dashboard JS to use forecast-api Cloud Run URL"
echo "   3. Test: curl <forecast-api-url>/health"
echo "============================================================"
