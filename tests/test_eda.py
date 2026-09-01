import pandas as pd

from src.eda import run_eda


def test_run_eda_writes_summary_and_charts(tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10),
            "aqi": range(40, 50),
            "pm25": range(10, 20),
            "pm10": range(20, 30),
            "temperature": range(25, 35),
            "humidity": range(50, 60),
        }
    )
    summary = run_eda(frame, output_dir=tmp_path)

    assert int(summary.loc[summary["metric"] == "rows", "value"].iloc[0]) == 10
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "missingness.csv").exists()
    assert (tmp_path / "correlation.csv").exists()
    assert (tmp_path / "aqi_trend.png").exists()
    assert (tmp_path / "distributions.png").exists()
    assert (tmp_path / "correlation.png").exists()
