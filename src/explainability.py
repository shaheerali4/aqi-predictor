from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

from .utils import setup_logging


logger = setup_logging(__name__)


def get_underlying_model(model):
    return getattr(model, "named_steps", {}).get("model", model)


def explain_row(
    model,
    row: pd.DataFrame,
    feature_names: list[str],
    background: pd.DataFrame | None = None,
    output_path: Path | None = None,
):
    row = row[feature_names]
    base_model = get_underlying_model(model)
    if shap is None:
        return pd.DataFrame({"feature": feature_names, "contribution": np.zeros(len(feature_names))})
    try:
        background = background[feature_names] if background is not None and not background.empty else row
        explainer = shap.Explainer(base_model.predict, background.values)
        explanation = explainer(row.values)
        values = explanation.values[0]
        result = pd.DataFrame({"feature": feature_names, "contribution": values})
        if output_path is not None:
            plt.figure(figsize=(8, 4))
            ordered = result.sort_values("contribution", key=np.abs, ascending=True)
            plt.barh(ordered["feature"], ordered["contribution"])
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
        return result.sort_values("contribution", key=np.abs, ascending=False)
    except Exception as exc:  # pragma: no cover - explainability should not break app
        logger.warning("SHAP explanation failed: %s", exc)
        return pd.DataFrame({"feature": feature_names, "contribution": np.zeros(len(feature_names))})
