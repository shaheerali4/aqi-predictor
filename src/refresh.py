from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .api_client import OpenMeteoAirQualityClient, OpenMeteoClient, OpenMeteoForecastClient
from .config import load_config
from .data_processing import normalize_openmeteo_air_quality, normalize_openmeteo_archive
from .feature_pipeline import build_feature_dataset, save_feature_dataset
from .utils import parse_date, setup_logging


logger = setup_logging(__name__)


def refresh_recent_data(
    city: str | None = None,
    country: str | None = None,
    days: int = 30,
    persist: bool = True,
) -> pd.DataFrame:
    """Fetch recent history plus today's live values and rebuild features."""

    config = load_config(city=city, country=country)
    today = date.today()
    archive_start = today - timedelta(days=max(days, 8))
    archive_end = today - timedelta(days=1)

    air_client = OpenMeteoAirQualityClient()
    weather_client = OpenMeteoClient()
    live_weather_client = OpenMeteoForecastClient()

    pollution_payload = air_client.get_archive_hourly(
        latitude=config.latitude,
        longitude=config.longitude,
        start_date=archive_start,
        end_date=archive_end,
        timezone=config.timezone,
    )
    weather_payload = weather_client.get_archive_hourly(
        latitude=config.latitude,
        longitude=config.longitude,
        start_date=archive_start,
        end_date=archive_end,
        timezone=config.timezone,
    )
    live_pollution_payload = air_client.get_forecast_hourly(
        latitude=config.latitude,
        longitude=config.longitude,
        timezone=config.timezone,
        forecast_days=1,
    )
    live_weather_payload = live_weather_client.get_forecast_hourly(
        latitude=config.latitude,
        longitude=config.longitude,
        timezone=config.timezone,
        forecast_days=1,
    )

    pollution = pd.concat(
        [
            normalize_openmeteo_air_quality(
                pollution_payload, city=config.city, country=config.country
            ),
            normalize_openmeteo_air_quality(
                live_pollution_payload, city=config.city, country=config.country
            ),
        ],
        ignore_index=True,
    )
    weather = pd.concat(
        [
            normalize_openmeteo_archive(
                weather_payload, city=config.city, country=config.country
            ),
            normalize_openmeteo_archive(
                live_weather_payload, city=config.city, country=config.country
            ),
        ],
        ignore_index=True,
    )
    pollution = pollution.drop_duplicates(["date", "city", "country"], keep="last")
    weather = weather.drop_duplicates(["date", "city", "country"], keep="last")
    features = build_feature_dataset(pollution, weather)
    if features.empty:
        raise RuntimeError("Live API refresh returned no overlapping air-quality and weather data.")
    if persist:
        save_feature_dataset(features, merge_existing=True)
    logger.info("Refreshed %s recent feature rows for %s.", len(features), config.city)
    return features
