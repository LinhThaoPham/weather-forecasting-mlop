#!/bin/bash
# run_retrain.sh — Chạy script này sau khi SSH vào GCP VM
# Usage: bash run_retrain.sh

set -e

echo "=========================================="
echo "  Weather MLOps — Retrain Pipeline"
echo "=========================================="

# Clone hoặc pull code mới nhất
REPO_DIR="$HOME/weather-forecasting-mlop"
REPO_URL="https://github.com/LinhThaoPham/weather-forecasting-mlop.git"

if [ -d "$REPO_DIR" ]; then
    echo "📥 Pulling latest code..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo "📥 Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# Tạo virtualenv nếu chưa có
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtualenv..."
    python3 -m venv venv
fi

source venv/bin/activate

# Cài dependencies
echo "📦 Installing dependencies..."
pip install --quiet -r requirements.txt

# Download DB từ GCS về local (để train)
echo "☁ Downloading DB from GCS..."
mkdir -p data models/current
gsutil cp gs://weather-forecasting-494811-models/data/weather_forecast.db ./data/ || echo "⚠ No DB found"
gsutil cp gs://weather-forecasting-494811-models/models/current/* ./models/current/ || echo "⚠ No existing models"
gsutil cp gs://weather-forecasting-494811-models/models/registry.json ./models/ || echo "⚠ No registry"
gsutil cp gs://weather-forecasting-494811-models/models/training_history.json ./models/ || echo "⚠ No history"

# Chạy retrain
echo ""
echo "🔄 Starting Retrain Pipeline..."
PYTHONPATH="$REPO_DIR" \
USE_GCS=true \
USE_BIGQUERY=false \
ENABLE_VERTEX_METRICS=false \
GCS_BUCKET_NAME=weather-forecasting-494811-models \
python src/training/retrain_pipeline.py

echo ""
echo "✅ Retrain complete! Models uploaded to GCS automatically."
echo "   CI/CD will pick up new models on next push, or trigger deploy manually."
echo "   Models: Prophet (temp + humidity + wind + cloud) × 3 cities (Hanoi, Danang, HCM) + LSTM"
