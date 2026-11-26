from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_DIR = MODELS_DIR / "registry"


CITY_PRESETS: Dict[str, dict] = {
    "Karachi": {
        "country": "Pakistan",
        "latitude": 24.8607,
        "longitude": 67.0011,
        "timezone": "Asia/Karachi",
    },
    "Lahore": {
        "country": "Pakistan",
        "latitude": 31.5204,
        "longitude": 74.3587,
        "timezone": "Asia/Karachi",
    },
    "Islamabad": {
        "country": "Pakistan",
        "latitude": 33.6844,
        "longitude": 73.0479,
        "timezone": "Asia/Karachi",
    },
    "Delhi": {
        "country": "India",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "timezone": "Asia/Kolkata",
    },
}


@dataclass(frozen=True)
class CityConfig:
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str


@dataclass(frozen=True)
class AppConfig:
    openaq_api_key: str | None
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    openaq_radius_meters: int
    historical_start_date: str
    historical_end_date: str | None
    feature_store_backend: str
    model_registry_backend: str
    default_forecast_horizons: tuple[int, ...]


def resolve_city_config(city: str | None = None, country: str | None = None) -> CityConfig:
    selected_city = (city or os.getenv("CITY", "Karachi")).strip()
    preset_name = next(
        (
            preset_city
            for preset_city in CITY_PRESETS
            if preset_city.casefold() == selected_city.casefold()
        ),
        None,
    )
    preset = CITY_PRESETS.get(preset_name) if preset_name else None
    latitude = os.getenv("CITY_LATITUDE")
    longitude = os.getenv("CITY_LONGITUDE")
    timezone = os.getenv("CITY_TIMEZONE")
    if preset is not None:
        return CityConfig(
            city=preset_name,
            country=country or preset["country"],
            latitude=float(latitude or preset["latitude"]),
            longitude=float(longitude or preset["longitude"]),
            timezone=timezone or preset["timezone"],
        )
    if latitude is None or longitude is None:
        raise ValueError(
            f"No preset found for city '{selected_city}'. "
            "Set CITY_LATITUDE and CITY_LONGITUDE in your environment."
        )
    return CityConfig(
        city=selected_city,
        country=country or os.getenv("COUNTRY", "Pakistan"),
        latitude=float(latitude),
        longitude=float(longitude),
        timezone=timezone or os.getenv("CITY_TIMEZONE", "UTC"),
    )


def load_config(city: str | None = None, country: str | None = None) -> AppConfig:
    city_config = resolve_city_config(city=city, country=country)
    horizons = tuple(
        int(item.strip())
        for item in os.getenv("DEFAULT_FORECAST_HORIZONS", "1,2,3").split(",")
        if item.strip()
    )
    return AppConfig(
        openaq_api_key=os.getenv("OPENAQ_API_KEY") or None,
        city=city_config.city,
        country=city_config.country,
        latitude=city_config.latitude,
        longitude=city_config.longitude,
        timezone=city_config.timezone,
        openaq_radius_meters=int(os.getenv("OPENAQ_RADIUS_METERS", "25000")),
        historical_start_date=os.getenv("HISTORICAL_START_DATE", "2024-01-01"),
        historical_end_date=os.getenv("HISTORICAL_END_DATE") or None,
        feature_store_backend=os.getenv("FEATURE_STORE_BACKEND", "local"),
        model_registry_backend=os.getenv("MODEL_REGISTRY_BACKEND", "local"),
        default_forecast_horizons=horizons,
    )

