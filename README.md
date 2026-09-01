# 10Pearls AQI Predictor

End-to-end AQI forecasting project for Karachi with a lightweight, student-friendly MLOps flow:

- external API ingestion
- historical backfill
- feature engineering
- local feature store fallback
- model training and comparison
- local model registry
- 3-day AQI forecasting
- SHAP-based explainability
- hazardous AQI alerts
- Streamlit dashboard
- live API refresh
- reproducible EDA outputs
- serverless FastAPI endpoint
- public Streamlit dashboard
- GitHub Actions automation

Public Streamlit app: https://aqi-predictor-j2mafjiq8p2gtvsrcc4rg2.streamlit.app

## Overview

This project predicts AQI for the next 3 days for a selected city. The working implementation uses:

- Open-Meteo Air Quality API for air-quality history and recent data
- Open-Meteo Historical Weather API for weather history
- a local CSV-based feature store fallback
- a local joblib-based model registry

The project is intentionally simple and reliable rather than over-engineered.

## Problem Statement

Air-quality conditions change over time and are influenced by weather patterns, pollution trends, and seasonal behavior. The goal is to build a practical forecasting pipeline that can learn from historical observations and predict AQI for the next 1, 2, and 3 days.

## Objectives

- ingest real data from public APIs
- build daily training data without leakage
- engineer lag and rolling features
- store processed features locally in a reusable abstraction
- train multiple models and compare them chronologically
- register the best model per horizon
- generate 3-day forecasts
- show explainability and alerts in a dashboard

## Architecture

1. `src/backfill.py` fetches historical air-quality and weather data.
2. `src/feature_pipeline.py` merges, cleans, and engineers features.
3. `src/feature_store.py` writes and reads the local feature store.
4. `src/train.py` trains 1-day, 2-day, and 3-day AQI models.
5. `src/model_registry.py` saves the best models and their metadata.
6. `src/refresh.py` refreshes recent history and today's live API values.
7. `src/predict.py` loads the latest features and produces forecasts.
8. `app/streamlit_app.py` renders the dashboard and explanations.
9. `api/index.py` exposes serverless-ready health and forecast endpoints.

## Project Structure

```text
AQI-Predictor/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── __init__.py
│   ├── api_client.py
│   ├── backfill.py
│   ├── config.py
│   ├── data_processing.py
│   ├── eda.py
│   ├── explainability.py
│   ├── feature_pipeline.py
│   ├── feature_store.py
│   ├── init.py
│   ├── model_registry.py
│   ├── predict.py
│   ├── refresh.py
│   ├── train.py
│   └── utils.py
├── notebooks/
│   └── eda.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── feature_store/
├── models/
├── tests/
├── .github/workflows/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
└── LICENSE
```

The FastAPI endpoint in `api/index.py` exposes `/health` and
`/forecast?city=Karachi&live=true` for local testing or a future API host.

The Streamlit dashboard is deployed publicly on Streamlit Community Cloud from
`app/streamlit_app.py`.

## Data Sources

### Air quality

- Open-Meteo Air Quality API
- endpoint: `https://air-quality-api.open-meteo.com/v1/air-quality`

This API provides hourly pollutant values and United States AQI fields. The project aggregates those hourly values to daily records for modeling.

### Weather

- Open-Meteo Historical Weather API
- endpoint: `https://archive-api.open-meteo.com/v1/archive`

The project uses hourly temperature, humidity, pressure, wind speed, and wind direction, then aggregates them to daily means.

## Features

Time-based features:

- `year`
- `month`
- `day`
- `day_of_week`
- `is_weekend`
- `week_of_year`

Lag and rolling features:

- lagged AQI
- lagged PM2.5 and PM10
- lagged weather variables
- 3-day rolling AQI mean
- 7-day rolling AQI mean
- AQI change and change rate

Targets:

- `target_aqi_tplus1`
- `target_aqi_tplus2`
- `target_aqi_tplus3`

## Machine Learning Models

The training pipeline compares:

- Random Forest Regressor
- Extra Trees Regressor
- Ridge Regression
- MLP neural network

The best model is chosen independently for each forecast horizon using chronological validation.

## Evaluation Metrics

- RMSE
- MAE
- R²

## Actual Model Results

Training was run successfully on a 91-row historical dataset for Karachi.

| Horizon | Best Model | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| 1 day | Random Forest | 5.5187 | 4.9846 | 0.7729 |
| 2 days | Random Forest | 9.9068 | 9.1470 | 0.2683 |
| 3 days | Ridge | 13.9966 | 12.2971 | -0.4605 |

## Feature Store

The project uses a local development feature store fallback:

- stored at `data/feature_store/features.csv`
- exposed through `src/feature_store.py`
- designed so a cloud feature store can be swapped in later

This is not Hopsworks or Vertex AI. It is a clean local abstraction for submission and learning.

## Model Registry

The registry is local and file-based:

- model artifacts stored in `models/`
- metadata stored in `models/registry/`

## Explainability

The project includes SHAP-based explainability in `src/explainability.py`. If a SHAP call is not suitable for a specific model/data combination, the code falls back safely rather than crashing the app.

The dashboard's `Refresh forecast` action fetches recent historical data plus today's
Open-Meteo values, rebuilds the feature row, and persists the update locally.

## AQI Alerts

The dashboard marks a hazardous alert when predicted AQI is at least 301.

## Dashboard

Run the app locally with:

```bash
streamlit run app/streamlit_app.py
```

The dashboard shows:

- latest observed AQI
- 3-day forecast table
- AQI category
- hazard flag
- recent AQI history
- model performance summary
- explainability table

## Automation

### Hourly feature pipeline

`.github/workflows/feature_pipeline.yml`

- scheduled hourly
- installs dependencies
- runs the backfill pipeline

### Daily training pipeline

`.github/workflows/training_pipeline.yml`

- scheduled daily
- installs dependencies
- runs the training pipeline

Both workflows upload generated data, EDA outputs, and model artifacts with GitHub
Actions artifact retention. The daily training workflow is self-contained: it
refreshes the training data before retraining.

## Setup Instructions

### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and adjust values if needed.

Current project defaults work for Karachi without any API key.

## Environment Variables

- `CITY`
- `COUNTRY`
- `CITY_LATITUDE`
- `CITY_LONGITUDE`
- `CITY_TIMEZONE`
- `HISTORICAL_START_DATE`
- `HISTORICAL_END_DATE`
- `FEATURE_STORE_BACKEND`
- `MODEL_REGISTRY_BACKEND`
- `DEFAULT_FORECAST_HORIZONS`
- `OPENAQ_API_KEY` optional, only if you later swap in OpenAQ

## How to Run

### Backfill

```bash
python -m src.backfill --start-date 2026-06-03 --end-date 2026-09-01
```

### Train

```bash
python -m src.train
```

### EDA

```bash
python -m src.eda
```

This writes summary tables and charts to `data/processed/eda/`. The source notebook
is `notebooks/eda.ipynb`.

### Predict

```bash
python -m src.predict
```

### Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Serverless API locally

```bash
uvicorn api.index:app --reload
```

Then open `http://127.0.0.1:8000/docs` or call `/forecast?live=true`.

### Public dashboard

The public dashboard is deployed on Streamlit Community Cloud:

https://aqi-predictor-j2mafjiq8p2gtvsrcc4rg2.streamlit.app

It runs `app/streamlit_app.py` directly and refreshes live data through the
Open-Meteo APIs.

## Tests

Run the unit tests with:

```bash
pytest -q
```

The current project passes all tests.

## Limitations

- the default model is trained on a relatively small daily dataset
- the local feature store and model registry remain the default development backends
- the project currently focuses on Karachi by default
- AQI is based on the available Open-Meteo fields and daily aggregation
- Streamlit Community Cloud deployment depends on the configured GitHub repository

## Future Improvements

- add more cities
- expand the backfill window where the data source allows it
- plug in a cloud feature store
- plug in a cloud model registry
- compare with more forecasting models
- add deployment monitoring
- add richer explainability charts
