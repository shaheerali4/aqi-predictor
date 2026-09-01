from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from .api_client import APIError, OpenMeteoAirQualityClient, OpenMeteoClient
from .config import load_config
from .data_processing import normalize_openmeteo_air_quality, normalize_openmeteo_archive
from .feature_pipeline import build_feature_dataset, save_feature_dataset
from .utils import calculate_aqi, parse_date, setup_logging


logger = setup_logging(__name__)


def choose_location_with_sensors(openaq_payload: dict) -> dict | None:
    for location in openaq_payload.get("results", []):
        sensors = location.get("sensors", []) or []
        parameter_names = {
            str(sensor.get("parameter", {}).get("name", "")).lower(): sensor
            for sensor in sensors
        }
        if "pm25" in parameter_names or "pm10" in parameter_names:
            location["sensor_map"] = parameter_names
            return location
    return None


def fetch_pollution_history(
    client: OpenMeteoAirQualityClient,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    city: str,
    country: str,
) -> pd.DataFrame:
    payload = client.get_archive_hourly(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        timezone="auto",
    )
    pollution = normalize_openmeteo_air_quality(payload, city=city, country=country)
    if pollution.empty:
        raise APIError("Open-Meteo did not return any hourly air-quality data.")
    return pollution


def fetch_weather_history(
    client: OpenMeteoClient,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str,
    city: str,
    country: str,
) -> pd.DataFrame:
    payload = client.get_archive_hourly(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
    )
    return normalize_openmeteo_archive(payload, city=city, country=country)


def build_backfill_dataset(city: str | None, country: str | None, start_date: str, end_date: str | None):
    config = load_config(city=city, country=country)
    start = parse_date(start_date) or parse_date(config.historical_start_date)
    end = parse_date(end_date) if end_date else parse_date(config.historical_end_date) or date.today()
    if start is None or end is None:
        raise ValueError("Both start and end dates must be available.")
    if end < start:
        raise ValueError("end-date must be on or after start-date.")

    openaq = OpenMeteoAirQualityClient()
    meteo = OpenMeteoClient()
    pollution = fetch_pollution_history(
        openaq,
        latitude=config.latitude,
        longitude=config.longitude,
        start_date=start,
        end_date=end,
        city=config.city,
        country=config.country,
    )
    weather = fetch_weather_history(
        meteo,
        latitude=config.latitude,
        longitude=config.longitude,
        start_date=start,
        end_date=end,
        timezone=config.timezone,
        city=config.city,
        country=config.country,
    )
    features = build_feature_dataset(pollution, weather)
    if features.empty or len(features) < 30:
        raise RuntimeError(
            "Insufficient historical observations for reliable training. "
            "Try expanding HISTORICAL_START_DATE or using a city with better coverage."
        )
    save_feature_dataset(features)
    return features


def main():
    parser = argparse.ArgumentParser(description="Build the AQI historical backfill dataset.")
    parser.add_argument("--city", default=None)
    parser.add_argument("--country", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    config = load_config(city=args.city, country=args.country)
    start_date = args.start_date or config.historical_start_date
    end_date = args.end_date or config.historical_end_date
    features = build_backfill_dataset(
        city=config.city,
        country=config.country,
        start_date=start_date,
        end_date=end_date,
    )
    logger.info("Backfill complete with %s rows.", len(features))


if __name__ == "__main__":
    main()
