import pandas as pd

from src.data_processing import add_lag_features, build_supervised_frame
from src.utils import aqi_category, calculate_aqi


def test_calculate_aqi_uses_pm25_and_pm10():
    assert calculate_aqi(pm25=35.0, pm10=50.0) is not None


def test_aqi_category():
    assert aqi_category(25) == "Good"
    assert aqi_category(175) == "Unhealthy"


def test_build_supervised_frame_creates_targets():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "city": ["Karachi"] * 10,
            "country": ["Pakistan"] * 10,
            "pm25": range(10, 20),
            "pm10": range(20, 30),
            "temperature": range(30, 40),
            "humidity": range(40, 50),
            "pressure": range(50, 60),
            "wind_speed": range(60, 70),
            "wind_direction": range(70, 80),
            "aqi": range(10, 20),
        }
    )
    out = build_supervised_frame(df)
    assert "target_aqi_tplus1" in out.columns
    assert "aqi_lag_1" in out.columns

