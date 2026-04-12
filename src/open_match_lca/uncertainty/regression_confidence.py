from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_CONFIDENCE_FEATURE_SPECS: tuple[tuple[str, float], ...] = (
    ("top1_probability", 1.0),
    ("top1_top2_margin", 1.0),
    ("retrieval_score_gap", 1.0),
    ("score_entropy", -1.0),
    ("interval_width", -1.0),
    ("factor_weighted_std", -1.0),
    ("top1_top2_hierarchy_distance", -1.0),
)


def _safe_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index, dtype=float)
    return frame[column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_std(values: pd.Series) -> float:
    std = float(values.std(ddof=0))
    return std if std > 1e-12 else 1.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_regression_confidence_calibrator(
    frame: pd.DataFrame,
    error_col: str = "error",
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot fit confidence calibrator from an empty frame.")
    if error_col not in frame.columns:
        raise ValueError(f"Frame must include {error_col} column to fit confidence calibrator.")

    errors = _safe_series(frame, error_col)
    target = -errors
    feature_entries: list[dict[str, float | str]] = []
    raw_weights: list[float] = []

    for feature_name, direction in DEFAULT_CONFIDENCE_FEATURE_SPECS:
        values = _safe_series(frame, feature_name)
        mean = float(values.mean())
        std = _safe_std(values)
        oriented = direction * values
        corr = spearmanr(oriented, target).statistic
        weight = float(0.0 if corr is None or np.isnan(corr) else max(float(corr), 0.0))
        raw_weights.append(weight)
        feature_entries.append(
            {
                "name": feature_name,
                "direction": float(direction),
                "mean": mean,
                "std": std,
            }
        )

    weight_sum = float(sum(raw_weights))
    if weight_sum <= 1e-12:
        uniform_weight = 1.0 / max(1, len(feature_entries))
        for entry in feature_entries:
            entry["weight"] = uniform_weight
    else:
        for entry, weight in zip(feature_entries, raw_weights, strict=False):
            entry["weight"] = float(weight / weight_sum)

    return {
        "version": 1,
        "features": feature_entries,
    }


def apply_regression_confidence_calibrator(
    frame: pd.DataFrame,
    calibrator: dict[str, Any] | None,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if not calibrator or not calibrator.get("features"):
        interval_width = _safe_series(frame, "interval_width")
        top1_probability = _safe_series(frame, "top1_probability")
        fallback = top1_probability / (interval_width + 1e-6)
        return fallback.astype(float)

    combined = np.zeros(len(frame), dtype=float)
    total_weight = 0.0
    for feature in calibrator["features"]:
        feature_name = str(feature["name"])
        direction = float(feature.get("direction", 1.0))
        mean = float(feature.get("mean", 0.0))
        std = float(feature.get("std", 1.0)) or 1.0
        weight = float(feature.get("weight", 0.0))
        values = _safe_series(frame, feature_name).to_numpy(dtype=float)
        z_values = direction * ((values - mean) / std)
        combined += weight * _sigmoid(z_values)
        total_weight += weight
    if total_weight <= 1e-12:
        return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index, dtype=float)
    return pd.Series(combined / total_weight, index=frame.index, dtype=float)
