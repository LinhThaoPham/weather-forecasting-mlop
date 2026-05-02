# src/data_pipeline/feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config.constants import DAILY_LAG_LIST, HOURLY_LAG_LIST, NUM_CITIES
from src.config.gcp import USE_BIGQUERY


def add_features(df: pd.DataFrame, is_hourly: bool = True) -> pd.DataFrame:
    """Add temporal and lag features for Prophet regressors."""
    df = df.copy()
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month

    if is_hourly:
        df["hour"] = df["ds"].dt.hour

    lag_list = HOURLY_LAG_LIST if is_hourly else DAILY_LAG_LIST

    for lag in lag_list:
        if "y" in df.columns:
            df[f"temp_lag_{lag}"] = df["y"].shift(lag)
        if "residual" in df.columns:
            df[f"residual_lag_{lag}"] = df["residual"].shift(lag)

    return df


def sliding_window(
    data: np.ndarray, window_size: int, forecast_horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for LSTM.

    Returns:
        X: shape (n_samples, window_size) or (n_samples, window_size, n_features)
        y: shape (n_samples, forecast_horizon) or (n_samples, forecast_horizon, n_features)
    """
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size : i + window_size + forecast_horizon])
    return np.array(X), np.array(y)


def normalize_data(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Normalize data to [0, 1] using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(data.reshape(-1, 1))
    return normalized.flatten(), scaler


def denormalize_data(data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse MinMax scaling."""
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()


# ============== Multi-City Features ==============


def build_multi_city_hourly() -> tuple[pd.DataFrame, list[str]]:
    """Build hourly temperature matrix from SQLite: (N_hours, 3 cities).

    Returns:
        df_pivot: DataFrame with columns = city_ids, index = timestamp (hourly)
        city_ids: ordered list of city IDs matching column order
    """
    from src.config.constants import LSTM_CITY_IDS

    city_ids = sorted(LSTM_CITY_IDS)

    if USE_BIGQUERY:
        from src.data_pipeline.bigquery_storage import fetch_historical_df
        df = fetch_historical_df()[["city_id", "timestamp", "temperature"]]
    else:
        from src.config.db import get_connection
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT city_id, timestamp, temperature
                   FROM weather_historical
                   ORDER BY timestamp, city_id"""
            ).fetchall()
        df = pd.DataFrame(rows, columns=["city_id", "timestamp", "temperature"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Pivot: rows=timestamps, columns=cities
    df_pivot = df.pivot(index="timestamp", columns="city_id", values="temperature")
    df_pivot = df_pivot[city_ids]  # enforce consistent column order
    df_pivot = df_pivot.ffill().bfill()  # fill gaps before dropping
    df_pivot = df_pivot.dropna()   # drop remaining incomplete rows
    df_pivot = df_pivot.sort_index()

    print(f"  📊 Multi-city hourly matrix: {df_pivot.shape[0]} hours × {df_pivot.shape[1]} cities")
    print(f"     Range: {df_pivot.index[0]} → {df_pivot.index[-1]}")
    print(f"     Cities: {city_ids}")

    return df_pivot, city_ids


def build_multi_city_daily() -> tuple[pd.DataFrame, list[str]]:
    """Build daily avg temperature matrix from SQLite: (N_days, 3 cities).

    Returns:
        df_pivot: DataFrame with columns = city_ids, index = date
        city_ids: ordered list of city IDs matching column order
    """
    from src.config.constants import LSTM_CITY_IDS

    city_ids = sorted(LSTM_CITY_IDS)

    if USE_BIGQUERY:
        from src.data_pipeline.bigquery_storage import fetch_historical_df
        raw_df = fetch_historical_df()[["city_id", "timestamp", "temperature"]]
        raw_df["date"] = pd.to_datetime(raw_df["timestamp"]).dt.date
        df = raw_df.groupby(["city_id", "date"], as_index=False)["temperature"].mean()
        df = df.rename(columns={"temperature": "temp_avg"})
    else:
        from src.config.db import get_connection
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT city_id,
                          DATE(timestamp) as date,
                          AVG(temperature) as temp_avg
                   FROM weather_historical
                   GROUP BY city_id, DATE(timestamp)
                   ORDER BY date, city_id"""
            ).fetchall()
        df = pd.DataFrame(rows, columns=["city_id", "date", "temp_avg"])

    df["date"] = pd.to_datetime(df["date"])

    # Pivot: rows=dates, columns=cities
    df_pivot = df.pivot(index="date", columns="city_id", values="temp_avg")
    df_pivot = df_pivot[city_ids]  # enforce consistent column order
    df_pivot = df_pivot.ffill().bfill()  # fill gaps before dropping
    df_pivot = df_pivot.dropna()   # drop remaining incomplete rows
    df_pivot = df_pivot.sort_index()

    print(f"  📊 Multi-city daily matrix: {df_pivot.shape[0]} days × {df_pivot.shape[1]} cities")
    print(f"     Date range: {df_pivot.index[0].date()} → {df_pivot.index[-1].date()}")
    print(f"     Cities: {city_ids}")

    return df_pivot, city_ids


def normalize_multi_city(
    data: np.ndarray,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Normalize multi-city data: (N, num_cities) → [0, 1] per-city.

    Uses one scaler that normalizes across all cities together.
    """
    if np.isnan(data).any():
        print(f"  ⚠ NaN detected in input ({np.isnan(data).sum()} values), replacing with column mean")
        col_means = np.nanmean(data, axis=0)
        for col in range(data.shape[1]):
            mask = np.isnan(data[:, col])
            data[mask, col] = col_means[col]
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(data)  # (N, num_cities) → (N, num_cities)
    return normalized, scaler


def denormalize_multi_city(
    data: np.ndarray, scaler: MinMaxScaler
) -> np.ndarray:
    """Inverse multi-city scaling."""
    return scaler.inverse_transform(data.reshape(-1, scaler.n_features_in_)).reshape(data.shape)
