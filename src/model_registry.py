from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODELS_DIR, REGISTRY_DIR
from .utils import ensure_dir, read_json, setup_logging, write_json


logger = setup_logging(__name__)


@dataclass
class ModelRecord:
    model_name: str
    horizon: int
    artifact_path: str
    metrics: dict[str, float]
    feature_columns: list[str]
    metadata: dict[str, Any]


class LocalModelRegistry:
    def __init__(self, registry_dir: Path | None = None):
        self.registry_dir = Path(registry_dir or REGISTRY_DIR)
        ensure_dir(self.registry_dir)
        ensure_dir(MODELS_DIR)

    def register(
        self,
        model,
        model_name: str,
        horizon: int,
        metrics: dict[str, float],
        feature_columns: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> ModelRecord:
        artifact_path = MODELS_DIR / f"{model_name}_h{horizon}.joblib"
        joblib.dump(model, artifact_path)
        record = ModelRecord(
            model_name=model_name,
            horizon=horizon,
            artifact_path=str(Path("models") / artifact_path.name),
            metrics=metrics,
            feature_columns=feature_columns,
            metadata=metadata or {},
        )
        write_json(self.registry_dir / f"{model_name}_h{horizon}.json", asdict(record))
        return record

    def list_records(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self.registry_dir.glob("*.json")):
            records.append(read_json(path))
        return records

    def load_model(self, model_name: str, horizon: int):
        artifact_path = MODELS_DIR / f"{model_name}_h{horizon}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model has not been trained yet. Missing artifact: {artifact_path}"
            )
        return joblib.load(artifact_path)

    def best_record(self, horizon: int) -> dict[str, Any] | None:
        candidates = [
            record
            for record in self.list_records()
            if record and int(record["horizon"]) == horizon
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item["metrics"]["rmse"])
