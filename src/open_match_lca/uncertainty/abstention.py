from __future__ import annotations

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
