from __future__ import annotations

import numpy as np
import pandas as pd

from open_match_lca.uncertainty.abstention import retained_risk, risk_coverage_curve


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
    retained_summary = retained_risk(frame, error_col="error", retained_col="retained")
    aurc = float(np.trapz(curve["retained_risk"], curve["coverage"])) if not curve.empty else 0.0
    eligible_curve = curve.loc[curve["coverage"] >= 0.5].copy()
    best_curve_gain = 0.0
    if not eligible_curve.empty:
        best_curve_gain = float(frame["error"].mean() - eligible_curve["retained_risk"].min())
    return {
        "empirical_coverage": empirical_coverage(frame, "y_true", "lower", "upper"),
        "average_interval_width": average_interval_width(frame, "lower", "upper"),
        "calibration_error": float((frame["confidence"] - calibration_target).abs().mean()),
        "retained_coverage": float(retained_summary["coverage"]),
        "retained_risk": float(retained_summary["retained_risk"]),
        "abstention_gain": float(retained_summary["abstention_gain"]),
        "best_curve_gain@50": best_curve_gain,
        "aurc": aurc,
    }
