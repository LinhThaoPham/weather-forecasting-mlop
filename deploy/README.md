# 🚀 GCP Cloud Run Deployment Guide

## Kiến trúc tổng quan

```
User → Cloud Load Balancer → Dashboard (Cloud Run)
                                ↓
                         Forecast API (Orchestrator)
                          ↙             ↘
                   Prophet API      LSTM API
                      ↓                ↓
                   GCS (Models)    GCS (Models)
                          ↘            ↙
                        Cloud SQL (PostgreSQL)
```

## Yêu cầu

- [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install)
- Docker Desktop (để build images locally)
- GCP Account với billing enabled (hoặc Free Trial $300)

## Quick Start (5 phút)

### 1. Đăng nhập GCP

```bash
gcloud auth login
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
```

### 2. Chạy script setup tự động

```bash
chmod +x deploy/setup_gcp.sh
./deploy/setup_gcp.sh YOUR_PROJECT_ID
```

Script sẽ tự động:
- ✅ Enable tất cả APIs
- ✅ Tạo Cloud SQL PostgreSQL
- ✅ Tạo GCS bucket + upload models
- ✅ Tạo Secrets
- ✅ Tạo Pub/Sub topics + Cloud Scheduler
- ✅ Build + Deploy tất cả services

### 3. Lấy URLs

```bash
# Xem tất cả service URLs
gcloud run services list --region=asia-southeast1
```

### 4. Test

```bash
# Health check
curl https://forecast-api-xxx-as.a.run.app/health

# Dự báo thời tiết
curl -X POST https://forecast-api-xxx-as.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "hanoi", "mode": "hourly", "hours": 72}'
```

## Deploy Manual (từng bước)

### Build & Push images

```bash
REGION=asia-southeast1
PROJECT_ID=$(gcloud config get-value project)

# Build từng service
docker build -f deploy/Dockerfile.data-api -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/data-api:latest .
docker build -f deploy/Dockerfile.forecast-api -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/forecast-api:latest .
docker build -f deploy/Dockerfile.prophet-api -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/prophet-api:latest .
docker build -f deploy/Dockerfile.lstm-api -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/lstm-api:latest .

# Push
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/data-api:latest
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/forecast-api:latest
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/prophet-api:latest
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/lstm-api:latest
```

### Deploy Cloud Run Services

```bash
# Data API
gcloud run deploy data-api \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/data-api:latest \
  --region=${REGION} \
  --allow-unauthenticated \
  --memory=512Mi --cpu=1 \
  --min-instances=1 --max-instances=50 \
  --cpu-boost

# Forecast API
gcloud run deploy forecast-api \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/forecast-api:latest \
  --region=${REGION} \
  --allow-unauthenticated \
  --memory=512Mi --cpu=1 \
  --min-instances=1 --max-instances=50 \
  --cpu-boost

# Prophet API
gcloud run deploy prophet-api \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/prophet-api:latest \
  --region=${REGION} \
  --memory=1Gi --cpu=2 \
  --min-instances=1 --max-instances=20 \
  --cpu-boost

# LSTM API
gcloud run deploy lstm-api \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/weather/lstm-api:latest \
  --region=${REGION} \
  --memory=2Gi --cpu=2 \
  --min-instances=1 --max-instances=10 \
  --cpu-boost
```

## Environment Variables

### Cloud Run Services

| Variable | Service | Description |
|----------|---------|-------------|
| `DB_TYPE` | data-api | `postgres` (Cloud SQL) hoặc `sqlite` (local) |
| `DB_USER` | data-api | PostgreSQL username |
| `DB_PASS` | data-api | PostgreSQL password |
| `DB_NAME` | data-api | Database name |
| `INSTANCE_CONNECTION_NAME` | data-api | `project:region:instance` |
| `USE_GCS` | prophet-api, lstm-api | `true` để load models từ GCS |
| `GCS_BUCKET` | prophet-api, lstm-api | GCS bucket name |
| `OPENWEATHER_API_KEY` | data-api, forecast-api | OWM API key |
| `PROPHET_API_URL` | forecast-api | URL của prophet-api Cloud Run |
| `LSTM_API_URL` | forecast-api | URL của lstm-api Cloud Run |
| `DATA_API_URL` | forecast-api | URL của data-api Cloud Run |

## Local Development

Vẫn chạy như cũ — không cần GCP:

```bash
# SQLite + local models (mặc định)
python -m uvicorn services.data_api.main:app --port 8001
python -m uvicorn services.prophet_api.main:app --port 8003
python -m uvicorn services.lstm_api.main:app --port 8004
python -m uvicorn services.forecast_api.main:app --port 8000

# Hoặc Docker Compose
docker-compose up
```

## Chi phí ước tính

| Resource | Monthly |
|----------|---------|
| Cloud Run (4 APIs, min=1) | ~$30-50 |
| Cloud SQL (db-f1-micro) | ~$10 |
| GCS (500MB) | ~$0.5 |
| Cloud Scheduler + Pub/Sub | Free tier |
| **Total** | **~$40-60** |

> 💡 GCP Free Trial: $300 credit / 90 ngày = ~5 tháng miễn phí!
