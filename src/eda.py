from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import PROCESSED_DIR
from .feature_pipeline import load_feature_dataset
from .utils import ensure_dir, setup_logging


logger = setup_logging(__name__)
EDA_DIR = PROCESSED_DIR / "eda"


def run_eda(df: pd.DataFrame | None = None, output_dir: Path | None = None) -> pd.DataFrame:
    """Create reproducible EDA tables and charts from the feature dataset."""

    frame = df.copy() if df is not None else load_feature_dataset()
    if frame.empty:
        raise RuntimeError("Feature dataset is empty. Run the backfill pipeline first.")
    frame = frame.sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"])
    destination = ensure_dir(output_dir or EDA_DIR)

    numeric = frame.select_dtypes(include="number")
    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "start_date",
                "end_date",
                "aqi_mean",
                "aqi_median",
                "aqi_min",
                "aqi_max",
                "aqi_std",
            ],
            "value": [
                len(frame),
                frame["date"].min().date(),
                frame["date"].max().date(),
                frame["aqi"].mean(),
                frame["aqi"].median(),
                frame["aqi"].min(),
                frame["aqi"].max(),
                frame["aqi"].std(),
            ],
        }
    )
    summary.to_csv(destination / "summary.csv", index=False)
    frame.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_csv(
        destination / "missingness.csv"
    )
    numeric.corr(numeric_only=True).to_csv(destination / "correlation.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(frame["date"], frame["aqi"], color="#d1495b", linewidth=2)
    plt.title("Daily AQI Trend")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination / "aqi_trend.png", dpi=140)
    plt.close()

    columns = [column for column in ["aqi", "pm25", "pm10", "temperature", "humidity"] if column in frame]
    frame[columns].hist(figsize=(11, 7), bins=15, color="#00798c", edgecolor="white")
    plt.suptitle("AQI and Weather Distributions")
    plt.tight_layout()
    plt.savefig(destination / "distributions.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 7))
    plt.imshow(numeric.corr(numeric_only=True), cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(destination / "correlation.png", dpi=140)
    plt.close()

    logger.info("EDA outputs written to %s.", destination)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate AQI EDA tables and charts.")
    parser.add_argument("--output-dir", default=str(EDA_DIR))
    args = parser.parse_args()
    print(run_eda(output_dir=Path(args.output_dir)).to_string(index=False))


if __name__ == "__main__":
    main()
