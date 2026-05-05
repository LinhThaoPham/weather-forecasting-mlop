"""Model & Training constants — hyperparameters only.

API URLs, timeouts, data params → see settings.py
"""

# Data Pipeline
HISTORICAL_DAYS_HOURLY = 730
HISTORICAL_DAYS_DAILY = 1000
FORECAST_HOURS_DEFAULT = 72
DEFAULT_CITY = "hanoi"

# LSTM Architecture
LSTM_ENCODER_UNITS = 128
LSTM_DECODER_UNITS = 64
LSTM_DROPOUT_RATE = 0.2
LSTM_LEARNING_RATE = 0.001
LSTM_CITY_IDS = ["hanoi", "danang", "hcm"]  # Bắc-Trung-Nam
NUM_CITIES = 3  # feature_dim for multi-city LSTM

# LSTM Windows
HOURLY_LOOKBACK = 24
HOURLY_HORIZON = 72
DAILY_LOOKBACK = 90    # 90 days lookback (1 season, balances quality vs sample count)
DAILY_HORIZON = 3

# Training
TRAIN_SPLIT_RATIO = 0.8
MAX_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10

# Ensemble Weights
PROPHET_WEIGHT = 0.6
LSTM_WEIGHT = 0.4

# Multi-variable prediction targets
# temperature = LSTM + Prophet ensemble, others = Prophet only
WEATHER_TARGETS = ["temperature", "humidity", "wind_speed", "cloud_cover"]
PROPHET_EXTRA_TARGETS = ["humidity", "wind_speed", "cloud_cover", "precipitation"]

# Evaluation — Champion/Challenger
# Model mới phải tốt hơn (MAE ≤ cũ), RMSE không tệ hơn 5%
RMSE_TOLERANCE = 0.05

# Monitoring — Performance tracking
DRIFT_THRESHOLD = 2.0               # °C — MAE trung bình vượt ngưỡng này = cảnh báo
SUSTAINED_DRIFT_DAYS = 7            # Số ngày liên tiếp vượt threshold mới trigger retrain
RETRAIN_COOLDOWN_DAYS = 14          # Minimum days giữa 2 lần retrain

# Retrain
RETRAIN_INTERVAL_DAYS = 90          # Seasonal retrain (mỗi 3 tháng)

# Feature Engineering
HOURLY_LAG_LIST = [1, 2, 3, 6, 12, 24]
DAILY_LAG_LIST = [1, 2, 3, 7]


def model_filename(model_type: str, mode: str, city: str = "", target: str = "") -> str:
    """Generate consistent model filename. ASCII-safe.

    Prophet = per-city (city required), LSTM = multi-city (no city suffix).
    target: for multi-variable Prophet (e.g. 'humidity', 'wind_speed').
           Empty or 'temperature' = default (backward compatible).
    """
    ext = "json" if model_type == "prophet" else "h5"
    target_suffix = f"_{target}" if target and target != "temperature" else ""
    if city:
        return f"{model_type}_{mode}_{city}{target_suffix}.{ext}"
    return f"{model_type}_{mode}{target_suffix}.{ext}"


def scaler_filename(mode: str) -> str:
    """Generate scaler filename for multi-city LSTM."""
    return f"lstm_{mode}_scaler.pkl"
