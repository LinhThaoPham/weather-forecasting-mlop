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

# Evaluation
MAE_REGRESSION_THRESHOLD = 0.10

# Retrain
RETRAIN_INTERVAL_DAYS = 90

# Feature Engineering
HOURLY_LAG_LIST = [1, 2, 3, 6, 12, 24]
DAILY_LAG_LIST = [1, 2, 3, 7]


def model_filename(model_type: str, mode: str, city: str = "") -> str:
    """Generate consistent model filename. ASCII-safe.

    Prophet = per-city (city required), LSTM = multi-city (no city suffix).
    """
    ext = "json" if model_type == "prophet" else "h5"
    if city:
        return f"{model_type}_{mode}_{city}.{ext}"
    return f"{model_type}_{mode}.{ext}"


def scaler_filename(mode: str) -> str:
    """Generate scaler filename for multi-city LSTM."""
    return f"lstm_{mode}_scaler.pkl"
