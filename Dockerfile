# syntax=docker/dockerfile:1

# Stage 1: build dependencies into an isolated virtualenv
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m venv ${VENV_PATH} \
    && ${VENV_PATH}/bin/pip install --no-cache-dir --upgrade pip \
    && ${VENV_PATH}/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: minimal runtime image
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# curl is required by docker-compose healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "--version"]
