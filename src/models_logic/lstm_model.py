# src/models_logic/lstm_model.py
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

from src.config.constants import (
    EARLY_STOP_PATIENCE,
    LSTM_DECODER_UNITS,
    LSTM_DROPOUT_RATE,
    LSTM_ENCODER_UNITS,
    LSTM_LEARNING_RATE,
)


class LSTMWeatherModel:
    """LSTM encoder-decoder for time series forecasting."""

    def __init__(self, lookback_window: int = 24, forecast_horizon: int = 72, feature_dim: int = 1):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.feature_dim = feature_dim
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def build_model(self) -> keras.Model:
        """Build encoder-decoder LSTM architecture."""
        inputs = keras.Input(shape=(self.lookback_window, self.feature_dim))

        encoder = layers.LSTM(LSTM_ENCODER_UNITS, activation="relu")(inputs)
        encoder = layers.Dropout(LSTM_DROPOUT_RATE)(encoder)
        encoder = layers.Dense(LSTM_DECODER_UNITS, activation="relu")(encoder)

        decoder_input = layers.RepeatVector(self.forecast_horizon)(encoder)
        decoder = layers.LSTM(LSTM_DECODER_UNITS, activation="relu", return_sequences=True)(decoder_input)
        decoder = layers.Dropout(LSTM_DROPOUT_RATE)(decoder)
        outputs = layers.TimeDistributed(layers.Dense(self.feature_dim))(decoder)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE),
            loss="mse",
            metrics=["mae"],
        )
        return self.model

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using MinMax scaler."""
        return self.scaler.fit_transform(data.reshape(-1, 1))

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize predictions."""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM."""
        from src.data_pipeline.feature_engineering import sliding_window
        return sliding_window(data, self.lookback_window, self.forecast_horizon)

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32):
        """Train with early stopping."""
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        )
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> keras.Model:
        """Load model from disk.

        Uses compile=False to avoid metric deserialization issues across
        Keras versions, then recompiles with the canonical optimizer/loss.
        """
        self.model = keras.models.load_model(path, compile=False)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE),
            loss="mse",
            metrics=["mae"],
        )
        print(f"Model loaded from {path}")
        return self.model


def create_lstm_model(
    lookback_window: int = 24,
    forecast_horizon: int = 72,
    feature_dim: int = 1,
) -> LSTMWeatherModel:
    """Factory function to create and build LSTM model.

    Args:
        lookback_window: number of timesteps to look back
        forecast_horizon: number of timesteps to predict
        feature_dim: number of input features (1=univariate, 6=multi-city)
    """
    model = LSTMWeatherModel(
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        feature_dim=feature_dim,
    )
    model.build_model()
    return model
