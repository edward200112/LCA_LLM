from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage_curve(frame: pd.DataFrame, confidence_col: str, error_col: str) -> pd.DataFrame:
    ranked = frame.sort_values(confidence_col, ascending=False).reset_index(drop=True)
    rows = []
    for retained in range(1, len(ranked) + 1):
        subset = ranked.iloc[:retained]
        rows.append(
            {
                "coverage": retained / len(ranked),
                "retained_risk": float(subset[error_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def retained_risk(
    frame: pd.DataFrame,
    error_col: str,
    retained_col: str = "retained",
) -> dict[str, float]:
    if frame.empty:
        raise ValueError("Cannot compute retained risk from an empty frame.")
    errors = frame[error_col].astype(float)
    baseline_risk = float(errors.mean())
    if retained_col not in frame.columns:
        return {
            "coverage": 1.0,
            "retained_risk": baseline_risk,
            "baseline_risk": baseline_risk,
            "abstention_gain": 0.0,
        }
    retained_mask = frame[retained_col].astype(bool)
    if retained_mask.sum() == 0:
        return {
            "coverage": 0.0,
            "retained_risk": baseline_risk,
            "baseline_risk": baseline_risk,
            "abstention_gain": 0.0,
        }
    selected_risk = float(errors.loc[retained_mask].mean())
    return {
        "coverage": float(retained_mask.mean()),
        "retained_risk": selected_risk,
        "baseline_risk": baseline_risk,
        "abstention_gain": float(baseline_risk - selected_risk),
    }


def learn_abstention_threshold(
    frame: pd.DataFrame,
    confidence_col: str,
    error_col: str,
    num_candidates: int = 21,
    min_coverage: float = 0.5,
    coverage_weight: float = 0.5,
) -> float:
    if frame.empty:
        raise ValueError("Cannot learn abstention threshold from an empty frame.")
    confidences = frame[confidence_col].astype(float).to_numpy()
    errors = frame[error_col].astype(float).to_numpy()
    baseline_error = float(np.mean(errors))
    best_threshold = float(confidences.min())
    best_score = 0.0
    for quantile in np.linspace(0.0, 1.0, num_candidates):
        threshold = float(np.quantile(confidences, quantile))
        retained = errors[confidences >= threshold]
        if retained.size == 0:
            continue
        coverage = retained.size / errors.size
        if coverage < min_coverage:
            continue
        retained_error = float(np.mean(retained))
        gain = baseline_error - retained_error
        score = gain * (coverage ** coverage_weight)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold
