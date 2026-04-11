from __future__ import annotations

from typing import Iterable

import pandas as pd

from open_match_lca.regression.topk_factor_mixture import topk_factor_mixture


def build_factor_lookup(epa_factors: pd.DataFrame) -> dict[str, float]:
    required = {"naics_code", "factor_value"}
    missing = sorted(required - set(epa_factors.columns))
    if missing:
        raise ValueError(
            f"epa_factors is missing required columns: {missing}. "
            f"Available columns: {list(epa_factors.columns)}"
        )
    lookup_series = epa_factors.groupby("naics_code")["factor_value"].mean()
    return {str(code): float(value) for code, value in lookup_series.items()}


def _prepare_truth_lookup(split_frame: pd.DataFrame) -> dict[str, float | None]:
    if "factor_value" in split_frame.columns:
        return {
            str(row.product_id): (None if pd.isna(row.factor_value) else float(row.factor_value))
            for row in split_frame.itertuples(index=False)
        }
    return {str(row.product_id): None for row in split_frame.itertuples(index=False)}


def top1_factor_lookup_predictions(
    retrieval_records: Iterable[dict],
    factor_lookup: dict[str, float],
    split_frame: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    truth_lookup = _prepare_truth_lookup(split_frame)
    rows = []
    for record in retrieval_records:
        top_candidate = record.get("candidates", [None])[0] if record.get("candidates") else None
        predicted_naics = str(top_candidate.get("naics_code")) if top_candidate else None
        predicted_factor = factor_lookup.get(predicted_naics or "", None)
        rows.append(
            {
                "product_id": record["product_id"],
                "gold_naics_code": record.get("gold_naics_code"),
                "pred_naics_code": predicted_naics,
                "pred_factor_value": predicted_factor,
                "factor_baseline": model_name,
                "y_true": truth_lookup.get(str(record["product_id"])),
                "retrieval_score_top1": None if top_candidate is None else float(top_candidate["score"]),
            }
        )
    return pd.DataFrame(rows)


def topk_factor_mixture_predictions(
    retrieval_records: Iterable[dict],
    factor_lookup: dict[str, float],
    split_frame: pd.DataFrame,
    top_k: int,
    model_name: str,
) -> pd.DataFrame:
    truth_lookup = _prepare_truth_lookup(split_frame)
    rows = []
    for record in retrieval_records:
        candidates = record.get("candidates", [])[:top_k]
        filtered = [
            candidate
            for candidate in candidates
            if str(candidate.get("naics_code")) in factor_lookup
        ]
        if not filtered:
            rows.append(
                {
                    "product_id": record["product_id"],
                    "gold_naics_code": record.get("gold_naics_code"),
                    "pred_naics_code": None,
                    "pred_factor_value": None,
                    "factor_baseline": model_name,
                    "y_true": truth_lookup.get(str(record["product_id"])),
                    "topk_count": 0,
                    "prob_max": None,
                    "factor_mean": None,
                    "factor_std": None,
                    "factor_min": None,
                    "factor_max": None,
                }
            )
            continue
        factor_values = [factor_lookup[str(candidate["naics_code"])] for candidate in filtered]
        scores = [float(candidate["score"]) for candidate in filtered]
        mixture = topk_factor_mixture(factor_values, scores)
        rows.append(
            {
                "product_id": record["product_id"],
                "gold_naics_code": record.get("gold_naics_code"),
                "pred_naics_code": str(filtered[0]["naics_code"]),
                "pred_factor_value": mixture["prediction"],
                "factor_baseline": model_name,
                "y_true": truth_lookup.get(str(record["product_id"])),
                "topk_count": len(filtered),
                "prob_max": mixture["prob_max"],
                "factor_mean": mixture["factor_mean"],
                "factor_std": mixture["factor_std"],
                "factor_min": mixture["factor_min"],
                "factor_max": mixture["factor_max"],
            }
        )
    return pd.DataFrame(rows)
