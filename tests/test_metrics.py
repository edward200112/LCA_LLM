from __future__ import annotations

import pandas as pd

from open_match_lca.eval.eval_regression import compute_regression_metrics
from open_match_lca.eval.eval_retrieval import compute_retrieval_metrics
from open_match_lca.eval.eval_uncertainty import evaluate_uncertainty


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


def test_uncertainty_metrics() -> None:
    frame = pd.DataFrame(
        [
            {"y_true": 1.0, "lower": 0.8, "upper": 1.2, "confidence": 0.9, "error": 0.1, "correct": 1.0},
            {"y_true": 2.0, "lower": 1.7, "upper": 2.4, "confidence": 0.6, "error": 0.2, "correct": 1.0},
            {"y_true": 3.0, "lower": 3.2, "upper": 3.6, "confidence": 0.3, "error": 0.5, "correct": 0.0},
        ]
    )
    metrics = evaluate_uncertainty(frame)
    assert 0.0 <= metrics["empirical_coverage"] <= 1.0
    assert metrics["average_interval_width"] > 0.0
