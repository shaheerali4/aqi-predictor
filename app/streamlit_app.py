from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Streamlit Cloud executes this file from the app directory, so make the
# repository root importable before loading the shared application modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.explainability import explain_row
from src.feature_pipeline import load_feature_dataset
from src.predict import forecast_three_days
from src.refresh import refresh_recent_data
from src.utils import aqi_category, hazardous_alert


st.set_page_config(page_title="10Pearls AQI Predictor", layout="wide")


def load_model_summary():
    path = Path("data/processed/model_summary.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


st.title("10Pearls AQI Predictor")
st.caption("3-day AQI forecasting with a local feature store fallback and direct models.")

config = load_config()
city = st.sidebar.text_input("City", value=config.city)
country = st.sidebar.text_input("Country", value=config.country)

if st.sidebar.button("Refresh forecast"):
    with st.spinner("Fetching the latest air-quality and weather data..."):
        try:
            refresh_recent_data(city=city, country=country, persist=True)
            st.cache_data.clear()
            st.sidebar.success("Live data refreshed.")
        except Exception as exc:
            st.sidebar.error(f"Refresh failed: {exc}")

try:
    forecast, latest_row = forecast_three_days(city=city)
    latest_aqi = float(latest_row["aqi"].iloc[0]) if "aqi" in latest_row.columns else None
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest observed AQI", f"{latest_aqi:.1f}" if latest_aqi is not None else "N/A")
    col2.metric("AQI Category", aqi_category(latest_aqi))
    col3.metric("Hazard Alert", "Yes" if hazardous_alert(latest_aqi) else "No")

    st.subheader("3-Day Forecast")
    st.dataframe(forecast, use_container_width=True)

    st.subheader("Model Performance")
    summary = load_model_summary()
    if summary.empty:
        st.info("Run the training pipeline to populate model metrics.")
    else:
        st.dataframe(summary, use_container_width=True)

    st.subheader("Explainability")
    registry_path = Path("models/registry")
    if registry_path.exists():
        from src.model_registry import LocalModelRegistry

        registry = LocalModelRegistry()
        record = registry.best_record(1)
        if record:
            model = registry.load_model(record["model_name"], 1)
            background = load_feature_dataset().sort_values("date").tail(30)
            explanation = explain_row(
                model,
                latest_row,
                record["feature_columns"],
                background=background,
            )
            st.dataframe(explanation.head(10), use_container_width=True)
        else:
            st.info("Train the models first to see SHAP-based explanations.")

    st.subheader("Recent Data")
    history = load_feature_dataset()
    if not history.empty:
        history = history.sort_values("date").tail(90)
        st.line_chart(history.set_index("date")["aqi"])
    else:
        st.info("No historical data found yet. Run the backfill pipeline first.")
except Exception as exc:
    st.error(str(exc))
    st.info("Run `python -m src.backfill` and `python -m src.train` to enable the dashboard.")
