from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .config import load_config
from .feature_pipeline import get_feature_columns, get_latest_feature_row, load_feature_dataset
from .feature_store import get_feature_store
from .model_registry import LocalModelRegistry
from .utils import aqi_category, hazardous_alert, setup_logging


logger = setup_logging(__name__)


def build_prediction_input(city: str | None = None) -> pd.DataFrame:
    config = load_config(city=city)
    store = get_feature_store()
    latest = store.latest_features(config.city)
    if latest.empty:
        history = load_feature_dataset()
        if not history.empty and "city" in history.columns:
            history = history[history["city"].str.lower() == config.city.lower()]
        latest = get_latest_feature_row(history)
    if latest.empty:
        raise RuntimeError("Model has not been trained yet or no feature data is available.")
    return latest


def forecast_from_latest(latest: pd.DataFrame):
    registry = LocalModelRegistry()
    feature_columns = [column for column in get_feature_columns(latest) if not column.startswith("target_aqi_")]
    if not feature_columns:
        raise RuntimeError("No usable feature columns are available for prediction.")
    row = latest.iloc[-1:]
    forecast_rows = []
    for horizon in (1, 2, 3):
        record = registry.best_record(horizon)
        if record is None:
            raise FileNotFoundError(
                "Model has not been trained yet. Run the training pipeline first."
            )
        model = registry.load_model(record["model_name"], horizon)
        prediction = float(model.predict(row[record["feature_columns"]])[0])
        forecast_date = pd.to_datetime(row["date"].iloc[0]) + timedelta(days=horizon)
        forecast_rows.append(
            {
                "forecast_date": forecast_date.date(),
                "horizon_days": horizon,
                "predicted_aqi": prediction,
                "category": aqi_category(prediction),
                "hazardous_alert": hazardous_alert(prediction),
            }
        )
    return pd.DataFrame(forecast_rows), row


def forecast_three_days(city: str | None = None):
    config = load_config(city=city)
    latest = build_prediction_input(city=config.city)
    return forecast_from_latest(latest)


def main():
    forecast, _ = forecast_three_days()
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
