# Universal Dockerfile for all Weather MLOps services
# Deploy different services via --command flag on Cloud Run
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY services/ ./services/
COPY daily_pipeline.py .

# Create dirs for models and data
RUN mkdir -p /app/models/current /app/models/archive /app/data

# Non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Default: Forecast API (overridden per service)
CMD ["uvicorn", "services.forecast_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
