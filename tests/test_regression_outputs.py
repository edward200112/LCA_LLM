from __future__ import annotations

import pandas as pd

from open_match_lca.regression.baseline_factor_lookup import (
    build_factor_lookup,
    top1_factor_lookup_predictions,
    topk_factor_mixture_predictions,
)
from open_match_lca.regression.topk_factor_mixture import topk_factor_mixture


def test_topk_factor_mixture_output_shape() -> None:
    output = topk_factor_mixture([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert set(output.keys()) == {
        "prediction",
        "prob_max",
        "factor_mean",
        "factor_std",
        "factor_min",
        "factor_max",
    }
    assert output["factor_min"] == 1.0


def test_factor_lookup_baselines() -> None:
    epa = pd.DataFrame(
        [
            {"naics_code": "111111", "factor_value": 1.0},
            {"naics_code": "222222", "factor_value": 3.0},
        ]
    )
    split_frame = pd.DataFrame(
        [
            {"product_id": "p1", "factor_value": 1.0},
            {"product_id": "p2", "factor_value": 3.0},
        ]
    )
    retrieval_records = [
        {
            "product_id": "p1",
            "gold_naics_code": "111111",
            "candidates": [
                {"naics_code": "111111", "score": 2.0},
                {"naics_code": "222222", "score": 1.0},
            ],
        },
        {
            "product_id": "p2",
            "gold_naics_code": "222222",
            "candidates": [
                {"naics_code": "222222", "score": 2.0},
                {"naics_code": "111111", "score": 1.0},
            ],
        },
    ]
    factor_lookup = build_factor_lookup(epa)
    top1 = top1_factor_lookup_predictions(retrieval_records, factor_lookup, split_frame, "top1")
    topk = topk_factor_mixture_predictions(retrieval_records, factor_lookup, split_frame, 2, "mix")
    assert top1["pred_factor_value"].tolist() == [1.0, 3.0]
    assert topk["pred_factor_value"].notna().all()
