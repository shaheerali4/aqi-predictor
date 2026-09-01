import pandas as pd

from src.feature_store import LocalFeatureStore


def test_local_feature_store_roundtrip(tmp_path):
    store = LocalFeatureStore(storage_path=tmp_path / "features.csv")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "city": ["Karachi"],
            "country": ["Pakistan"],
            "aqi": [42.0],
        }
    )
    store.write_features(frame)
    latest = store.latest_features("Karachi")
    assert len(latest) == 1
    assert latest.iloc[0]["aqi"] == 42.0

