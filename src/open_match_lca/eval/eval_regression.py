from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-12, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def compute_regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    spearman = spearmanr(true, pred).statistic
    return {
        "mae": float(np.mean(np.abs(true - pred))),
        "rmse": rmse,
        "mape": mean_absolute_percentage_error(true, pred),
        "smape": symmetric_mape(true, pred),
        "spearman": float(0.0 if np.isnan(spearman) else spearman),
    }
