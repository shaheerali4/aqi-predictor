from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

import requests

from .utils import setup_logging


logger = setup_logging(__name__)


class APIError(RuntimeError):
    pass


@dataclass
class APIResponse:
    data: dict[str, Any]
    status_code: int


class BaseAPIClient:
    def __init__(self, base_url: str, timeout: int = 30, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(
                url,
                params=params or {},
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise APIError(f"Request to {url} timed out.") from exc
        except requests.RequestException as exc:
            raise APIError(f"Unable to fetch data from {url}: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise APIError(f"Invalid JSON returned by {url}") from exc
        if not isinstance(payload, dict):
            raise APIError(f"Unexpected response shape from {url}")
        return payload


class OpenAQClient(BaseAPIClient):
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        super().__init__("https://api.openaq.org", timeout=timeout, api_key=api_key)

    def find_locations(self, latitude: float, longitude: float, radius: int = 25000, limit: int = 10):
        return self.get_json(
            "/v3/locations",
            params={
                "coordinates": f"{latitude},{longitude}",
                "radius": radius,
                "limit": limit,
            },
        )

    def get_location(self, location_id: int):
        return self.get_json(f"/v3/locations/{location_id}")

    def get_location_latest(self, location_id: int):
        return self.get_json(f"/v3/locations/{location_id}/latest", params={"limit": 100})

    def get_sensor_daily(self, sensor_id: int, start_date: date, end_date: date):
        return self.get_json(
            f"/v3/sensors/{sensor_id}/days",
            params={
                "date_from": start_date.isoformat(),
                "date_to": end_date.isoformat(),
                "limit": 1000,
            },
        )


class OpenMeteoClient(BaseAPIClient):
    def __init__(self, timeout: int = 30):
        super().__init__("https://archive-api.open-meteo.com", timeout=timeout)

    def get_archive_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        timezone: str,
    ):
        return self.get_json(
            "/v1/archive",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timezone": timezone,
                "hourly": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "surface_pressure",
                        "wind_speed_10m",
                        "wind_direction_10m",
                    ]
                ),
            },
        )


class OpenMeteoForecastClient(BaseAPIClient):
    def __init__(self, timeout: int = 30):
        super().__init__("https://api.open-meteo.com", timeout=timeout)

    def get_forecast_hourly(
        self,
        latitude: float,
        longitude: float,
        timezone: str,
        forecast_days: int = 1,
    ):
        return self.get_json(
            "/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone,
                "forecast_days": forecast_days,
                "hourly": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "surface_pressure",
                        "wind_speed_10m",
                        "wind_direction_10m",
                    ]
                ),
            },
        )


class OpenMeteoAirQualityClient(BaseAPIClient):
    def __init__(self, timeout: int = 30):
        super().__init__("https://air-quality-api.open-meteo.com", timeout=timeout)

    def get_archive_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        timezone: str,
    ):
        return self.get_json(
            "/v1/air-quality",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timezone": timezone,
                "hourly": ",".join(
                    [
                        "pm2_5",
                        "pm10",
                        "carbon_monoxide",
                        "nitrogen_dioxide",
                        "sulphur_dioxide",
                        "ozone",
                        "us_aqi",
                    ]
                ),
            },
        )

    def get_forecast_hourly(
        self,
        latitude: float,
        longitude: float,
        timezone: str,
        forecast_days: int = 1,
    ):
        return self.get_json(
            "/v1/air-quality",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone,
                "forecast_days": forecast_days,
                "hourly": ",".join(
                    [
                        "pm2_5",
                        "pm10",
                        "carbon_monoxide",
                        "nitrogen_dioxide",
                        "sulphur_dioxide",
                        "ozone",
                        "us_aqi",
                    ]
                ),
            },
        )
