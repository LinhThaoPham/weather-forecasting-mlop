# src/models_logic/hybrid_ensemble.py
import numpy as np

from src.config.constants import LSTM_WEIGHT, PROPHET_WEIGHT


def ensemble_predict(
    prophet_predictions: np.ndarray | None,
    lstm_predictions: np.ndarray | None,
    prophet_weight: float = PROPHET_WEIGHT,
    lstm_weight: float = LSTM_WEIGHT,
) -> np.ndarray:
    """Blend Prophet and LSTM predictions by weight.

    Strategy:
        - Prophet (60%): Strong seasonal patterns (yearly, weekly, daily)
        - LSTM (40%): Short-term temporal patterns & non-linearities

    Graceful fallback: if one model fails, use 100% of the other.
    """
    if lstm_predictions is None:
        return prophet_predictions
    if prophet_predictions is None:
        return lstm_predictions

    min_len = min(len(prophet_predictions), len(lstm_predictions))
    prophet_pred = np.array(prophet_predictions[:min_len])
    lstm_pred = np.array(lstm_predictions[:min_len])

    return prophet_weight * prophet_pred + lstm_weight * lstm_pred