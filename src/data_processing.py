from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import calculate_aqi, setup_logging
from .utils import aqi_category as map_aqi_category


logger = setup_logging(__name__)

POLLUTANT_COLUMNS = ["pm25", "pm10", "co", "no2", "so2", "o3"]
WEATHER_COLUMNS = ["temperature", "humidity", "pressure", "wind_speed", "wind_direction"]


def normalize_openaq_daily_response(payload: dict, city: str, country: str) -> pd.DataFrame:
    rows = []
    for result in payload.get("results", []):
        parameter = str(result.get("parameter", {}).get("name", "")).lower()
        if parameter not in {"pm25", "pm10", "co", "no2", "so2", "o3"}:
            continue
        dt = result.get("period", {}).get("datetimeTo", {}) or result.get("datetime", {})
        local_dt = dt.get("local") or dt.get("utc")
        rows.append(
            {
                "date": pd.to_datetime(local_dt).date() if local_dt else None,
                "city": city,
                "country": country,
                "parameter": parameter,
                "value": result.get("value"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["date"])
    return df


def pivot_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pivot = (
        df.pivot_table(
            index=["date", "city", "country"],
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values("date")
    )
    pivot.columns.name = None
    return pivot


def normalize_openmeteo_archive(payload: dict, city: str, country: str) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times),
            "temperature": hourly.get("temperature_2m"),
            "humidity": hourly.get("relative_humidity_2m"),
            "pressure": hourly.get("surface_pressure"),
            "wind_speed": hourly.get("wind_speed_10m"),
            "wind_direction": hourly.get("wind_direction_10m"),
            "city": city,
            "country": country,
        }
    )
    df["date"] = df["timestamp"].dt.date
    daily = (
        df.groupby(["date", "city", "country"], as_index=False)[WEATHER_COLUMNS]
        .mean(numeric_only=True)
        .sort_values("date")
    )
    return daily


def normalize_openmeteo_air_quality(payload: dict, city: str, country: str) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times),
            "pm25": hourly.get("pm2_5"),
            "pm10": hourly.get("pm10"),
            "co": hourly.get("carbon_monoxide"),
            "no2": hourly.get("nitrogen_dioxide"),
            "so2": hourly.get("sulphur_dioxide"),
            "o3": hourly.get("ozone"),
            "aqi": hourly.get("us_aqi"),
            "city": city,
            "country": country,
        }
    )
    df["date"] = df["timestamp"].dt.date
    daily = (
        df.groupby(["date", "city", "country"], as_index=False)[
            ["pm25", "pm10", "co", "no2", "so2", "o3", "aqi"]
        ]
        .mean(numeric_only=True)
        .sort_values("date")
    )
    if "aqi" not in daily.columns or daily["aqi"].isna().all():
        daily["aqi"] = [
            calculate_aqi(pm25=row.get("pm25"), pm10=row.get("pm10"))
            for _, row in daily.iterrows()
        ]
    return daily


def add_aqi_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["aqi"] = [
        calculate_aqi(pm25=row.get("pm25"), pm10=row.get("pm10"))
        for _, row in out.iterrows()
    ]
    out["aqi_category"] = out["aqi"].apply(map_aqi_category)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    return out


def add_lag_features(df: pd.DataFrame, columns: Iterable[str] | None = None, lags=(1, 2, 3)) -> pd.DataFrame:
    columns = list(columns or ["aqi", "pm25", "pm10", "temperature", "humidity", "pressure", "wind_speed"])
    out = df.sort_values("date").copy()
    for column in columns:
        if column not in out.columns:
            continue
        for lag in lags:
            out[f"{column}_lag_{lag}"] = out[column].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    for column in ["aqi", "pm25"]:
        if column in out.columns:
            out[f"{column}_roll_3"] = out[column].shift(1).rolling(3).mean()
            out[f"{column}_roll_7"] = out[column].shift(1).rolling(7).mean()
    if "aqi" in out.columns:
        out["aqi_change"] = out["aqi"] - out["aqi"].shift(1)
        out["aqi_change_rate"] = out["aqi_change"] / out["aqi"].shift(1).replace(0, np.nan)
    return out


def build_supervised_frame(df: pd.DataFrame, horizons=(1, 2, 3)) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = out.sort_values("date").reset_index(drop=True)
    for horizon in horizons:
        out[f"target_aqi_tplus{horizon}"] = out["aqi"].shift(-horizon)
    out = out.dropna(subset=["aqi"])
    return out


def merge_air_and_weather(pollution_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    if pollution_df.empty:
        return weather_df.copy()
    if weather_df.empty:
        return pollution_df.copy()
    merged = pollution_df.merge(weather_df, on=["date", "city", "country"], how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    if "aqi" not in merged.columns:
        merged["aqi"] = [
            calculate_aqi(pm25=row.get("pm25"), pm10=row.get("pm10"))
            for _, row in merged.iterrows()
        ]
    return merged


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "city",
        "country",
        "aqi_category",
        "aqi",
        "target_aqi_tplus1",
        "target_aqi_tplus2",
        "target_aqi_tplus3",
    }
    return [column for column in df.columns if column not in excluded and pd.api.types.is_numeric_dtype(df[column])]
