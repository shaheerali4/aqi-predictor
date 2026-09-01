# 10Pearls AQI Predictor Project Report

## Introduction

This project builds an end-to-end AQI forecasting system for Karachi. The system predicts AQI for the next 1, 2, and 3 days using historical air-quality and weather data, a local feature store fallback, a local model registry, and a Streamlit dashboard.

The implementation is intentionally practical and beginner-friendly. The goal is to demonstrate the complete MLOps-style workflow without over-engineering the solution.

## Problem Statement

Air-quality conditions are strongly influenced by atmospheric conditions and pollutant trends. A forecasting system that can estimate AQI several days ahead is useful for public awareness, health planning, and early warning.

## Objectives

- ingest real external API data
- build a leakage-safe training dataset
- engineer time-series features
- create a feature store abstraction
- train and compare multiple models
- register the best model for each horizon
- produce a 3-day AQI forecast
- show explainability and alerts
- automate the pipeline with GitHub Actions
- support live API refresh
- expose a serverless-ready API endpoint

## Requirements

The project needed to provide:

- external API ingestion
- feature engineering
- feature store
- historical backfill
- EDA
- ML training and evaluation
- model registry
- automated feature and training pipelines
- dashboard
- 3-day forecasting
- explainability
- hazardous AQI alerts
- documentation and report

## System Architecture

1. `src/backfill.py` pulls historical air-quality and weather data.
2. `src/feature_pipeline.py` merges and engineers features.
3. `src/feature_store.py` stores processed features in a local CSV-based fallback.
4. `src/train.py` trains models for 1-day, 2-day, and 3-day AQI forecasts.
5. `src/model_registry.py` saves the best model artifacts and metadata.
6. `src/predict.py` loads the latest row and generates forecasts.
7. `src/refresh.py` combines recent archive data with today's live API values.
8. `app/streamlit_app.py` visualizes predictions, history, and explanations.
9. `api/index.py` exposes serverless-ready health and forecast endpoints.

## Data Sources

### Air Quality

Primary source:

- Open-Meteo Air Quality API
- `https://air-quality-api.open-meteo.com/v1/air-quality`

Why this source was chosen:

- free to use for this project
- no API key required
- provides hourly pollutant values
- includes United States AQI fields

### Weather

Primary source:

- Open-Meteo Historical Weather API
- `https://archive-api.open-meteo.com/v1/archive`

Why this source was chosen:

- free
- no API key required
- provides historical hourly weather data
- suitable for daily aggregation

### OpenAQ

An OpenAQ client was implemented initially, but the live API now requires an API key in this environment. The working project uses Open-Meteo instead so the submission runs end-to-end without a hidden credential dependency.

## Data Collection

The backfill pipeline was executed for Karachi using:

- start date: `2026-06-03`
- end date: `2026-09-01`

The result was:

- 91 daily rows in the processed training dataset

The data was then aligned chronologically and merged on date, city, and country.

## Data Cleaning

The pipeline performs the following cleaning steps:

- parse timestamps into dates
- sort chronologically
- aggregate hourly data to daily means
- drop rows without usable dates
- avoid future leakage by creating lag features with `shift`
- create targets using future AQI values only

## Feature Engineering

Time-based features:

- `year`
- `month`
- `day`
- `day_of_week`
- `is_weekend`
- `week_of_year`

Lag features:

- `aqi_lag_1`, `aqi_lag_2`, `aqi_lag_3`
- `pm25_lag_1`, `pm25_lag_2`, `pm25_lag_3`
- `pm10_lag_1`, `pm10_lag_2`, `pm10_lag_3`
- weather lags for temperature, humidity, pressure, and wind speed

Rolling features:

- 3-day and 7-day AQI rolling means
- 3-day and 7-day PM2.5 rolling means

Trend features:

- `aqi_change`
- `aqi_change_rate`

Targets:

- `target_aqi_tplus1`
- `target_aqi_tplus2`
- `target_aqi_tplus3`

## Historical Backfill

The backfill script:

- fetches hourly air-quality data
- fetches hourly weather data
- aggregates both to daily rows
- engineers features
- saves the processed dataset
- writes the result into the local feature store fallback

This is a legitimate historical dataset generated from public APIs rather than fabricated values.

## EDA

An EDA notebook was created at:

- [notebooks/eda.ipynb](./notebooks/eda.ipynb)

The notebook investigates summary statistics, missingness, AQI trend, distributions,
and numeric correlations. The `src.eda` runner also writes reproducible CSV summaries
and PNG charts under `data/processed/eda/`.

## Model Development

The project compares four model families:

- Random Forest Regressor
- Extra Trees Regressor
- Ridge Regression
- MLP neural network

The models are trained separately for each forecast horizon:

- 1 day ahead
- 2 days ahead
- 3 days ahead

Chronological train/validation/test splits are used so the future never leaks into training.

## Model Comparison

Model selection is based on validation RMSE, and the final test metrics are recorded after retraining the selected model on the combined train and validation data.

### Actual Results

| Horizon | Best Model | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| 1 day | Random Forest | 5.5187 | 4.9846 | 0.7729 |
| 2 days | Random Forest | 9.9068 | 9.1470 | 0.2683 |
| 3 days | Ridge | 13.9966 | 12.2971 | -0.4605 |

Interpretation:

- the 1-day model is reasonably strong for this small dataset
- the 2-day model remains usable but is less accurate
- the 3-day horizon is the hardest and needs more history or richer features to improve

## Evaluation Metrics

- RMSE
- MAE
- R²

These metrics were computed on a chronologically held-out test split.

## Best Model

The best horizon-specific models were:

- horizon 1: Random Forest
- horizon 2: Random Forest
- horizon 3: Ridge

The candidate set also includes a scaled MLP neural network. On the current small
dataset, the selected models above were more accurate, so the MLP remains in the
comparison rather than being registered as the production model.

Artifacts were saved in:

- `models/random_forest_h1.joblib`
- `models/random_forest_h2.joblib`
- `models/ridge_h3.joblib`

## Feature Store

Implemented as a local development fallback:

- `src/feature_store.py`
- storage file: `data/feature_store/features.csv`

This abstraction is intentionally simple and can later be replaced by a cloud feature store such as Hopsworks or Vertex AI.

## Model Registry

Implemented as a local file-based registry:

- `src/model_registry.py`
- metadata files in `models/registry/`

The registry stores:

- model name
- horizon
- metrics
- feature columns
- saved artifact path

## Prediction Pipeline

The prediction pipeline:

- loads the latest feature row
- loads the correct model for each horizon
- predicts AQI for the next 1, 2, and 3 days
- maps AQI to a category
- raises a hazardous alert if AQI is at least 301

### Actual Forecast Output

| Forecast Date | Horizon | Predicted AQI | Category | Hazardous |
|---|---:|---:|---:|---:|
| 2026-09-02 | 1 | 69.0904 | Moderate | No |
| 2026-09-03 | 2 | 79.2742 | Moderate | No |
| 2026-09-04 | 3 | 68.1461 | Moderate | No |

## Explainability

The dashboard includes a SHAP-based explanation path. In practice, the explainability helper uses a background sample from recent training rows so the output is meaningful instead of collapsing to zeros.

The explanation table highlights the most influential features for the current forecast.

## AQI Alerts

The project defines a hazardous alert for AQI values of 301 or higher.

This is intentionally simple and easy to understand for the assignment.

## Streamlit Dashboard

The dashboard is implemented in:

- [app/streamlit_app.py](./app/streamlit_app.py)

It shows:

- latest observed AQI
- AQI category
- hazard flag
- 3-day forecast table
- recent AQI trend
- model summary
- explanation table

The `Refresh forecast` action fetches recent archive data and today's live Open-Meteo
values before rebuilding the latest feature row.

## CI/CD Automation

### Feature pipeline

File:

- `.github/workflows/feature_pipeline.yml`

Schedule:

- hourly

Purpose:

- run the feature/backfill pipeline automatically

### Training pipeline

File:

- `.github/workflows/training_pipeline.yml`

Schedule:

- daily

Purpose:

- retrain and refresh the model artifacts automatically

Both workflows upload processed data, EDA outputs, and model artifacts using GitHub
Actions artifact retention. The daily training job performs its own backfill first,
so it does not depend on a previous runner's local filesystem.

## Serverless API

`api/index.py` provides `/health` and `/forecast` endpoints. Passing `live=true`
fetches current API values without writing to the filesystem, which is appropriate
for a serverless read-only runtime. `vercel.json` configures the Python function for
Vercel deployment.

## Testing

The test suite includes:

- AQI calculation checks
- feature generation checks
- feature store round-trip checks
- synthetic training check
- API error handling check

### Actual Test Result

Command:

```bash
pytest -q
```

Result:

- `6 passed`

## Deployment

The app runs locally with:

```bash
streamlit run app/streamlit_app.py
```

The API can be deployed to Vercel with the included `vercel.json`; actual deployment
requires a Vercel account and project credentials.

## Limitations

- the backfill window is relatively small
- the project currently focuses on Karachi by default
- the feature store is local rather than cloud-hosted
- the model registry is local rather than cloud-hosted
- the 3-day horizon is the least accurate
- more cities and a longer history would likely improve performance

## Future Improvements

- add more city presets
- expand the historical window
- add a cloud feature store implementation
- add a cloud model registry
- try stronger time-series models
- add richer SHAP charts
- deploy the dashboard publicly

## Conclusion

This project satisfies the core assignment requirements with a real, working pipeline:

- public API ingestion
- historical backfill
- feature engineering
- feature storage
- model training
- evaluation
- model registry
- forecasting
- explainability
- alerts
- dashboard
- automation

The implementation is intentionally practical and submission-ready rather than overly complex.
