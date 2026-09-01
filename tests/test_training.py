import pandas as pd

from src.train import train_horizon_model


def test_train_horizon_model_with_synthetic_data():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "city": ["Karachi"] * 60,
            "country": ["Pakistan"] * 60,
            "pm25": range(60),
            "pm10": range(60, 120),
            "temperature": range(10, 70),
            "humidity": range(30, 90),
            "pressure": range(1000, 1060),
            "wind_speed": range(1, 61),
            "wind_direction": range(90, 150),
            "aqi": range(100, 160),
            "year": [2024] * 60,
            "month": [1] * 60,
            "day": list(range(1, 31)) * 2,
            "day_of_week": [i % 7 for i in range(60)],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1] * 8 + [0, 0, 0, 0],
            "week_of_year": [1] * 60,
            "aqi_lag_1": range(99, 159),
            "aqi_lag_2": range(98, 158),
            "aqi_lag_3": range(97, 157),
            "pm25_lag_1": range(59, 119),
            "pm10_lag_1": range(119, 179),
            "temperature_lag_1": range(9, 69),
            "humidity_lag_1": range(29, 89),
            "pressure_lag_1": range(999, 1059),
            "wind_speed_lag_1": range(0, 60),
            "aqi_roll_3": range(100, 160),
            "aqi_roll_7": range(100, 160),
            "pm25_roll_3": range(60),
            "pm25_roll_7": range(60),
            "aqi_change": [1] * 60,
            "aqi_change_rate": [0.01] * 60,
            "target_aqi_tplus1": range(101, 161),
            "target_aqi_tplus2": range(102, 162),
            "target_aqi_tplus3": range(103, 163),
        }
    )
    best_name, model, feature_columns, results, metrics = train_horizon_model(df, 1)
    assert best_name
    assert feature_columns
    assert "rmse" in metrics
