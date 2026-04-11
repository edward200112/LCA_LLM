from __future__ import annotations

import numpy as np


def conformal_quantile(residuals: list[float], alpha: float = 0.1) -> float:
    if not residuals:
        raise ValueError("residuals must be non-empty for conformal calibration")
    array = np.sort(np.asarray(residuals, dtype=float))
    index = int(np.ceil((1 - alpha) * (len(array) + 1))) - 1
    index = max(0, min(index, len(array) - 1))
    return float(array[index])


def apply_conformal_interval(predictions: list[float], qhat: float) -> list[dict[str, float]]:
    return [
        {"lower": float(pred - qhat), "upper": float(pred + qhat)}
        for pred in predictions
    ]
