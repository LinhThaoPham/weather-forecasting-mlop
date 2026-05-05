"""Model Evaluation — Compute metrics and compare models for accept/rollback decisions."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config.constants import RMSE_TOLERANCE


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE for predictions."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Remove NaN/inf pairs
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid_mask.any():
        return {"mae": 999.0, "rmse": 999.0, "mape": 999.0}
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    nonzero_mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        if nonzero_mask.any()
        else 0.0
    )

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 2),
    }


def compare_models(
    new_metrics: dict,
    old_metrics: dict | None,
) -> str:
    """Champion/Challenger: new model must be BETTER, not just 'acceptable'.

    Accept only if:
      1. MAE ≤ old (must improve or tie)
      2. RMSE not worse by more than RMSE_TOLERANCE (5%)
    """
    if old_metrics is None:
        return "accept"

    old_mae = old_metrics.get("mae", float("inf"))
    new_mae = new_metrics.get("mae", float("inf"))

    # Guard: skip comparison if old metrics are invalid
    if old_mae == 0 or old_mae == float("inf"):
        return "accept"

    # Condition 1: MAE must not increase
    if new_mae > old_mae:
        pct = (new_mae - old_mae) / old_mae * 100
        print(f"⚠ Model regression: MAE {old_mae:.4f} → {new_mae:.4f} (+{pct:.1f}%) → ROLLBACK")
        return "rollback"

    # Condition 2: RMSE must not degrade beyond tolerance
    old_rmse = old_metrics.get("rmse", float("inf"))
    new_rmse = new_metrics.get("rmse", float("inf"))
    if old_rmse > 0 and old_rmse != float("inf"):
        if new_rmse > old_rmse * (1 + RMSE_TOLERANCE):
            print(f"⚠ RMSE regression: {old_rmse:.4f} → {new_rmse:.4f} (>{RMSE_TOLERANCE*100:.0f}% tolerance) → ROLLBACK")
            return "rollback"

    improvement = (old_mae - new_mae) / old_mae * 100
    print(f"✓ Model improved: MAE {old_mae:.4f} → {new_mae:.4f} (-{improvement:.1f}%)")
    return "accept"


def evaluate_prophet(model, df_test, mode: str = "hourly") -> dict:
    """Evaluate Prophet model on test data."""
    try:
        # Prophet.predict() only requires 'ds'; it auto-creates needed columns
        future_df = df_test[["ds"]].copy()
        forecast = model.predict(future_df)
        y_true = df_test["y"].values
        y_pred = forecast["yhat"].values
        min_len = min(len(y_true), len(y_pred))
        return compute_metrics(y_true[:min_len], y_pred[:min_len])
    except (KeyError, ValueError) as e:
        print(f"⚠ Prophet evaluation error: {e}")
        return {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}


def evaluate_lstm(model, X_test: np.ndarray, y_test: np.ndarray, scaler) -> dict:
    """Evaluate LSTM model on test data (univariate)."""
    try:
        predictions = model.predict(X_test)
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return compute_metrics(y_true, y_pred)
    except (ValueError, TypeError) as e:
        print(f"⚠ LSTM evaluation error: {e}")
        return {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}


def evaluate_lstm_multi_city(model, X_test: np.ndarray, y_test: np.ndarray, scaler) -> dict:
    """Evaluate LSTM model on test data (multi-city).

    Denormalizes all cities together, then flattens for MAE/RMSE/MAPE.
    """
    try:
        predictions = model.predict(X_test)
        n_features = scaler.n_features_in_

        # Replace NaN in predictions with 0 before inverse_transform
        if np.isnan(predictions).any():
            nan_count = int(np.isnan(predictions).sum())
            print(f"  ⚠ {nan_count} NaN values in LSTM predictions, replacing with 0")
            predictions = np.nan_to_num(predictions, nan=0.0)

        # Reshape to 2D for scaler: (N * horizon, n_cities)
        y_true_2d = y_test.reshape(-1, n_features)
        y_pred_2d = predictions.reshape(-1, n_features)

        # Also guard y_true
        y_true_2d = np.nan_to_num(y_true_2d, nan=0.0)

        y_true_real = scaler.inverse_transform(y_true_2d).flatten()
        y_pred_real = scaler.inverse_transform(y_pred_2d).flatten()

        return compute_metrics(y_true_real, y_pred_real)
    except (ValueError, TypeError) as e:
        print(f"⚠ LSTM multi-city evaluation error: {e}")
        return {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}

