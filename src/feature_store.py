from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import FEATURE_STORE_DIR
from .utils import ensure_dir, setup_logging


logger = setup_logging(__name__)


class LocalFeatureStore:
    """Simple local fallback feature store using CSV storage."""

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = Path(storage_path or FEATURE_STORE_DIR / "features.csv")
        ensure_dir(self.storage_path.parent)

    def write_features(self, df: pd.DataFrame) -> Path:
        if df.empty:
            raise ValueError("Cannot write an empty feature frame.")
        existing = self.read_features()
        combined = pd.concat([existing, df], ignore_index=True) if not existing.empty else df.copy()
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values(["city", "date"]).drop_duplicates(
                subset=["city", "date"], keep="last"
            )
        combined.to_csv(self.storage_path, index=False)
        return self.storage_path

    def read_features(
        self,
        city: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not self.storage_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.storage_path, parse_dates=["date"])
        if city:
            df = df[df["city"].str.lower() == city.lower()]
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        return df.reset_index(drop=True)

    def latest_features(self, city: str, rows: int = 1) -> pd.DataFrame:
        df = self.read_features(city=city)
        if df.empty:
            return df
        return df.sort_values("date").tail(rows).reset_index(drop=True)


def get_feature_store() -> LocalFeatureStore:
    return LocalFeatureStore()

