from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .predict import forecast_from_latest
from .refresh import refresh_recent_data
from .utils import ensure_dir


def export_forecast(path="forecast.json"):
    features = refresh_recent_data(persist=False)
    forecast, latest = forecast_from_latest(features)
    latest_aqi = latest["aqi"].iloc[-1] if "aqi" in latest.columns else None
    payload = {
        "city": str(latest["city"].iloc[-1]) if "city" in latest.columns else "Karachi",
        "country": str(latest["country"].iloc[-1]) if "country" in latest.columns else "Pakistan",
        "latest_observed_aqi": latest_aqi,
        "forecast": forecast.to_dict(orient="records"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(path)
    destination = ensure_dir(output_path.parent) / output_path.name
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(export_forecast())
