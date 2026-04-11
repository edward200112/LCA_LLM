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


def learn_abstention_threshold(
    frame: pd.DataFrame,
    confidence_col: str,
    error_col: str,
    num_candidates: int = 21,
) -> float:
    if frame.empty:
        raise ValueError("Cannot learn abstention threshold from an empty frame.")
    confidences = frame[confidence_col].astype(float).to_numpy()
    errors = frame[error_col].astype(float).to_numpy()
    baseline_error = float(np.mean(errors))
    best_threshold = float(confidences.min())
    best_score = -np.inf
    for quantile in np.linspace(0.0, 1.0, num_candidates):
        threshold = float(np.quantile(confidences, quantile))
        retained = errors[confidences >= threshold]
        if retained.size == 0:
            continue
        coverage = retained.size / errors.size
        retained_error = float(np.mean(retained))
        score = (baseline_error - retained_error) * coverage
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold
