from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR
from .feature_pipeline import get_feature_columns, load_feature_dataset
from .model_registry import LocalModelRegistry
from .utils import chronological_split, ensure_dir, setup_logging


logger = setup_logging(__name__)
TARGET_TEMPLATE = "target_aqi_tplus{horizon}"


def build_models():
    return {
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "mlp_neural_network": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        alpha=0.001,
                        max_iter=2000,
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
        "mae": float(mean_absolute_error(y_val, preds)),
        "r2": float(r2_score(y_val, preds)),
    }


def train_horizon_model(df: pd.DataFrame, horizon: int):
    target = TARGET_TEMPLATE.format(horizon=horizon)
    if target not in df.columns:
        raise ValueError(f"Missing target column {target}. Run the backfill first.")
    usable = df.dropna(subset=[target]).copy()
    feature_columns = get_feature_columns(usable)
    feature_columns = [column for column in feature_columns if not column.startswith("target_aqi_")]
    usable = usable.sort_values("date").reset_index(drop=True)
    X = usable[feature_columns]
    y = usable[target]
    train_df, val_df, test_df = chronological_split(usable)
    X_train, y_train = train_df[feature_columns], train_df[target]
    X_val, y_val = val_df[feature_columns], val_df[target]
    X_test, y_test = test_df[feature_columns], test_df[target]
    if len(train_df) < 10 or len(val_df) < 5 or len(test_df) < 5:
        raise RuntimeError("Not enough rows after chronological split to train and evaluate.")

    candidates = build_models()
    results = []
    best_name = None
    best_model = None
    best_val_rmse = np.inf
    for name, model in candidates.items():
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
        results.append({"model_name": name, **metrics})
        if metrics["rmse"] < best_val_rmse:
            best_val_rmse = metrics["rmse"]
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None
    best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_preds = best_model.predict(X_test)
    test_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "mae": float(mean_absolute_error(y_test, test_preds)),
        "r2": float(r2_score(y_test, test_preds)),
    }
    return best_name, best_model, feature_columns, results, test_metrics


def train_all_horizons(city: str | None = None, country: str | None = None):
    df = load_feature_dataset()
    if df.empty:
        raise RuntimeError(
            "Feature dataset is empty. Run python -m src.backfill before training."
        )
    registry = LocalModelRegistry()
    summary_rows = []
    for horizon in (1, 2, 3):
        best_name, best_model, feature_columns, comparison, test_metrics = train_horizon_model(df, horizon)
        artifact = registry.register(
            model=best_model,
            model_name=best_name,
            horizon=horizon,
            metrics=test_metrics,
            feature_columns=feature_columns,
            metadata={"comparison": comparison},
        )
        summary_rows.append(
            {
                "horizon": horizon,
                "best_model": best_name,
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "r2": test_metrics["r2"],
                "artifact_path": artifact.artifact_path,
            }
        )
        logger.info(
            "Horizon %s trained with %s | RMSE=%.3f MAE=%.3f R2=%.3f",
            horizon,
            best_name,
            test_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["r2"],
        )
    summary = pd.DataFrame(summary_rows)
    ensure_dir(PROCESSED_DIR)
    summary.to_csv(PROCESSED_DIR / "model_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train AQI forecasting models.")
    parser.add_argument("--city", default=None)
    parser.add_argument("--country", default=None)
    args = parser.parse_args()
    summary = train_all_horizons(city=args.city, country=args.country)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
