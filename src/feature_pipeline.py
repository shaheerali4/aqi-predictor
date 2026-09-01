from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR
from .data_processing import build_supervised_frame, merge_air_and_weather, select_feature_columns
from .feature_store import get_feature_store
from .utils import ensure_dir, setup_logging


logger = setup_logging(__name__)


def build_feature_dataset(pollution_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    merged = merge_air_and_weather(pollution_df, weather_df)
    if merged.empty:
        return merged
    features = build_supervised_frame(merged)
    return features


def save_feature_dataset(
    df: pd.DataFrame,
    filename: str = "training_data.csv",
    merge_existing: bool = False,
) -> Path:
    ensure_dir(PROCESSED_DIR)
    path = PROCESSED_DIR / filename
    output = df.copy()
    if merge_existing and path.exists():
        existing = load_feature_dataset(path)
        output = pd.concat([existing, output], ignore_index=True)
        output["date"] = pd.to_datetime(output["date"])
        output = output.sort_values(["city", "date"]).drop_duplicates(
            subset=["city", "date"], keep="last"
        )
    output.to_csv(path, index=False)
    get_feature_store().write_features(output if merge_existing else df)
    return path


def load_feature_dataset(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or PROCESSED_DIR / "training_data.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


def get_latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("date").tail(1).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return select_feature_columns(df)
