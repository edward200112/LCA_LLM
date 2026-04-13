from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


@dataclass(frozen=True)
class PairedTestResult:
    metric: str
    primary_model: str
    baseline_model: str
    n: int
    mean_primary: float
    std_primary: float
    mean_baseline: float
    std_baseline: float
    mean_difference: float
    paired_t_pvalue: float | None
    wilcoxon_pvalue: float | None


def mean_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Expected equal length inputs, got {len(left)} and {len(right)}")
    return float(np.mean(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def summarize_metric_by_model(
    frame: pd.DataFrame,
    model_col: str,
    metric_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for model_name, group in frame.groupby(model_col):
        for metric in metric_names:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "model": str(model_name),
                    "metric": metric,
                    "n": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def paired_significance_tests(
    frame: pd.DataFrame,
    model_col: str,
    seed_col: str,
    primary_model: str,
    baseline_models: list[str],
    metric_names: list[str],
) -> pd.DataFrame:
    results: list[dict[str, float | str | int | None]] = []
    primary_frame = frame.loc[frame[model_col] == primary_model].copy()
    if primary_frame.empty:
        raise ValueError(f"Primary model {primary_model} not found in frame.")

    for baseline_model in baseline_models:
        baseline_frame = frame.loc[frame[model_col] == baseline_model].copy()
        if baseline_frame.empty:
            continue
        merged = primary_frame.merge(
            baseline_frame,
            on=seed_col,
            suffixes=("_primary", "_baseline"),
            how="inner",
        )
        for metric in metric_names:
            primary_col = f"{metric}_primary"
            baseline_col = f"{metric}_baseline"
            if primary_col not in merged.columns or baseline_col not in merged.columns:
                continue
            paired = merged[[seed_col, primary_col, baseline_col]].dropna()
            if paired.empty:
                continue
            left = paired[primary_col].astype(float).to_numpy()
            right = paired[baseline_col].astype(float).to_numpy()
            t_pvalue = None
            wilcoxon_pvalue = None
            if paired.shape[0] >= 2:
                t_result = ttest_rel(left, right, nan_policy="omit")
                t_pvalue = None if np.isnan(t_result.pvalue) else float(t_result.pvalue)
                try:
                    w_result = wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")
                    wilcoxon_pvalue = None if np.isnan(w_result.pvalue) else float(w_result.pvalue)
                except ValueError:
                    wilcoxon_pvalue = None
            result = PairedTestResult(
                metric=metric,
                primary_model=primary_model,
                baseline_model=baseline_model,
                n=int(paired.shape[0]),
                mean_primary=float(np.mean(left)),
                std_primary=float(np.std(left, ddof=1)) if paired.shape[0] > 1 else 0.0,
                mean_baseline=float(np.mean(right)),
                std_baseline=float(np.std(right, ddof=1)) if paired.shape[0] > 1 else 0.0,
                mean_difference=float(np.mean(left - right)),
                paired_t_pvalue=t_pvalue,
                wilcoxon_pvalue=wilcoxon_pvalue,
            )
            results.append(result.__dict__)
    return pd.DataFrame(results)
