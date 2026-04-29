# src/models_logic/prophet_model.py
import json

from prophet import Prophet
from prophet.serialize import model_to_json


def train_prophet(df, is_hourly: bool = True) -> Prophet:
    """Train Prophet model for temperature forecasting."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=is_hourly,
    )
    model.add_regressor("humidity")
    model.add_regressor("cloud_cover")
    model.fit(df)
    return model


def save_prophet(model: Prophet, path: str) -> None:
    """Serialize Prophet model to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_to_json(model), f)