from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def setup_logging(name: str = "aqi_predictor") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def parse_date(value: str | date | datetime | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


def date_range(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def safe_mean(values: Iterable[float | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


AQI_BREAKPOINTS = {
    "pm25": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
}


def _sub_index(value: float | None, pollutant: str) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    for c_low, c_high, i_low, i_high in AQI_BREAKPOINTS[pollutant]:
        if c_low <= value <= c_high:
            return (i_high - i_low) / (c_high - c_low) * (value - c_low) + i_low
    return 500.0 if value > AQI_BREAKPOINTS[pollutant][-1][1] else None


def calculate_aqi(pm25: float | None = None, pm10: float | None = None) -> float | None:
    """Approximate AQI from the available particulate matter values.

    The project uses a pragmatic AQI estimate based on PM2.5/PM10 because
    those are the most consistently available pollutants in the free sources.
    """

    scores = [
        _sub_index(pm25, "pm25"),
        _sub_index(pm10, "pm10"),
    ]
    scores = [score for score in scores if score is not None]
    if not scores:
        return None
    return float(max(scores))


def aqi_category(aqi: float | None) -> str:
    if aqi is None or not np.isfinite(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def hazardous_alert(aqi: float | None) -> bool:
    return bool(aqi is not None and np.isfinite(aqi) and aqi >= 301)


def chronological_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = max(1, int(n * train_frac))
    val_end = max(train_end + 1, int(n * (train_frac + val_frac)))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def standardize_columns(df):
    return df.rename(columns={col: col.strip().lower() for col in df.columns})


def ensure_datetime_column(df, column: str = "date"):
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], utc=False)
    return df


def flatten_list(items: Sequence[Sequence]) -> list:
    return [value for seq in items for value in seq]
