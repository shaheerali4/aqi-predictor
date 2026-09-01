from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.predict import forecast_from_latest, forecast_three_days
from src.refresh import refresh_recent_data


app = FastAPI(title="10Pearls AQI Predictor API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "aqi-predictor"}


@app.get("/forecast")
def forecast(
    city: str = Query(default="Karachi"),
    country: str = Query(default="Pakistan"),
    live: bool = Query(default=False),
):
    try:
        if live:
            features = refresh_recent_data(city=city, country=country, persist=False)
            result, _ = forecast_from_latest(features)
        else:
            result, _ = forecast_three_days(city=city)
        return {
            "city": city,
            "country": country,
            "forecast": result.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
