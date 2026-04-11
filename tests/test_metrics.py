from __future__ import annotations

from open_match_lca.eval.eval_regression import compute_regression_metrics
from open_match_lca.eval.eval_retrieval import compute_retrieval_metrics


def test_retrieval_metrics() -> None:
    records = [
        {
            "gold_naics_code": "337127",
            "candidates": [
                {"naics_code": "337127", "score": 1.0},
                {"naics_code": "337211", "score": 0.5},
            ],
        },
        {
            "gold_naics_code": "335139",
            "candidates": [
                {"naics_code": "337127", "score": 1.0},
                {"naics_code": "335139", "score": 0.9},
            ],
        },
    ]
    metrics = compute_retrieval_metrics(records)
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["recall@5"] == 1.0
    assert metrics["hierarchical_accuracy@2"] == 1.0


def test_regression_metrics() -> None:
    metrics = compute_regression_metrics([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])
    assert metrics["mae"] > 0.0
    assert 0.0 <= metrics["smape"] <= 1.0
