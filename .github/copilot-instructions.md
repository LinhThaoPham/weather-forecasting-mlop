# Copilot instructions for `weather-forecasting-mlop`

## Build, test, and lint

### Environment setup
- Install dependencies: `python -m pip install -r requirements.txt`
- Ensure project root is on `PYTHONPATH` when running services/scripts directly.

### Build / run
- Start the local multi-service stack on Windows: `.\start_local.ps1`
- Build and run with Docker Compose: `docker compose up -d --build`
- Run the daily data+retrain pipeline once: `python daily_pipeline.py`
- Run retraining pipeline once: `python src/training/retrain_pipeline.py`
- Run retraining scheduler daemon: `python src/training/retrain_pipeline.py --daemon`

### Tests
- Run full suite: `pytest`
- Run a single test: `pytest tests/test_api.py::TestForecastAPI::test_health_endpoint -v`

### Lint
- No repository lint command/config is currently defined (`ruff`, `flake8`, `pylint`, `mypy`, etc. are not configured).

## High-level architecture

- The system is a Python microservice architecture with shared core code in `src/` and service entrypoints in `services/`.
- `services/forecast_api/main.py` is the **orchestrator**: it fetches base forecast features, calls Prophet and LSTM services via HTTP, then blends outputs (hybrid ensemble).
- `services/data_api/main.py` is the data-access layer around SQLite (`data/weather_forecast.db`) and external weather APIs; it initializes DB schema on startup.
- `services/prophet_api/main.py` and `services/lstm_api/main.py` are model-serving APIs that load artifacts from `models/current` and support hot-reload via `/reload`.
- Training/versioning is centralized in `src/training/retrain_pipeline.py`: train -> evaluate -> accept/rollback -> archive/deploy models -> reload model services -> update `models/registry.json`.
- The daily automation flow exists in both `daily_pipeline.py` and `.github/workflows/daily_pipeline.yml` (fetch latest weather data, retrain models, persist updated artifacts).

## Key conventions in this repo

- **Single source of truth for config**: use `src/config/settings.py` for service URLs, timeouts, API keys, DB path; use `src/config/cities.py` for city metadata.
- **Canonical city IDs** are lowercase keys (`hanoi`, `hcm`, `danang`, `haiphong`, `nhatrang`, `dalat`) and should stay consistent across APIs, training, and data pipeline.
- **Model artifact naming and locations matter**:
  - Artifact roots: `models/current`, `models/archive`, `models/registry.json`
  - Use filename helpers in `src/config/constants.py` (`model_filename`, `scaler_filename`) when adding/changing training outputs.
- **Service contract pattern**:
  - New clients should use `/forecast` on model services; `/predict_hourly` and `/predict_daily` remain for backward compatibility.
  - Retraining pipeline depends on `/reload` endpoints for hot-swapping models.
- **Data ingestion pattern**: preferred write path is `src/data_pipeline/store_data.py` (`store_historical`, `store_forecast`, `store_current`) with shared field mapping/timezone behavior.
- **Test expectations are integration-tolerant**: tests in `tests/test_api.py` frequently accept multiple status codes (`200/500/503`) because external APIs/model artifacts may be unavailable.
- **Current implementation note**: ensemble blending logic is duplicated in `src/models_logic/hybrid_ensemble.py` and `services/forecast_api/main.py`; keep both aligned unless refactoring to one shared implementation.
