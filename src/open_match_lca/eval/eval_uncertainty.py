from __future__ import annotations

import pandas as pd

from open_match_lca.uncertainty.abstention import risk_coverage_curve


def empirical_coverage(frame: pd.DataFrame, truth_col: str, lower_col: str, upper_col: str) -> float:
    covered = (
        (frame[truth_col] >= frame[lower_col]) & (frame[truth_col] <= frame[upper_col])
    ).mean()
    return float(covered)


def average_interval_width(frame: pd.DataFrame, lower_col: str, upper_col: str) -> float:
    return float((frame[upper_col] - frame[lower_col]).mean())


def calibration_error(frame: pd.DataFrame, confidence_col: str, correctness_col: str) -> float:
    return float((frame[confidence_col] - frame[correctness_col]).abs().mean())


def evaluate_uncertainty(frame: pd.DataFrame) -> dict[str, float]:
    curve = risk_coverage_curve(frame, "confidence", "error")
    calibration_target = 1.0 - frame["error"].rank(pct=True)
    return {
        "empirical_coverage": empirical_coverage(frame, "y_true", "lower", "upper"),
        "average_interval_width": average_interval_width(frame, "lower", "upper"),
        "calibration_error": float((frame["confidence"] - calibration_target).abs().mean()),
        "abstention_gain": float(curve["retained_risk"].iloc[0] - curve["retained_risk"].iloc[-1]),
    }
