from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from open_match_lca.regression.predict_regression import build_regression_feature_frame
from open_match_lca.regression.train_lgbm_regressor import train_lgbm_quantile_regressor
from open_match_lca.uncertainty.abstention import learn_abstention_threshold


def test_build_regression_feature_frame() -> None:
    retrieval_records = [
        {
            "product_id": "p1",
            "gold_naics_code": "337127",
            "query_text": "metal chair",
            "candidates": [
                {"naics_code": "337127", "score": 2.0},
                {"naics_code": "335139", "score": 1.0},
            ],
        },
        {
            "product_id": "p2",
            "gold_naics_code": "335139",
            "query_text": "office lamp",
            "candidates": [
                {"naics_code": "335139", "score": 2.0},
                {"naics_code": "337127", "score": 1.0},
            ],
        },
    ]
    products = pd.DataFrame(
        [
            {"product_id": "p1", "text": "metal chair", "gold_naics_code": "337127", "factor_value": 1.2, "text_len": 2, "has_numeric_tokens": False},
            {"product_id": "p2", "text": "office lamp", "gold_naics_code": "335139", "factor_value": 2.4, "text_len": 2, "has_numeric_tokens": False},
        ]
    )
    epa = pd.DataFrame(
        [
            {"naics_code": "337127", "factor_value": 1.2},
            {"naics_code": "335139", "factor_value": 2.4},
        ]
    )
    frame, projector = build_regression_feature_frame(
        retrieval_records=retrieval_records,
        products_frame=products,
        epa_factors=epa,
        top_k=2,
        pca_dim=2,
        fit_projector=True,
    )
    assert projector is not None
    assert "factor_weighted_mean" in frame.columns
    assert "text_pca_00" in frame.columns
    assert len(frame) == 2


def test_learn_abstention_threshold() -> None:
    frame = pd.DataFrame(
        [
            {"confidence": 0.9, "error": 0.1},
            {"confidence": 0.7, "error": 0.2},
            {"confidence": 0.2, "error": 1.0},
        ]
    )
    threshold = learn_abstention_threshold(frame, "confidence", "error")
    assert frame["confidence"].min() <= threshold <= frame["confidence"].max()


def test_train_lgbm_quantile_regressor_smoke(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    train_frame = pd.DataFrame(
        [
            {"product_id": "p1", "y_true": 1.0, "top1_probability": 0.9, "factor_weighted_mean": 1.1, "text_len": 2},
            {"product_id": "p2", "y_true": 2.0, "top1_probability": 0.8, "factor_weighted_mean": 1.9, "text_len": 3},
            {"product_id": "p3", "y_true": 3.0, "top1_probability": 0.7, "factor_weighted_mean": 2.8, "text_len": 4},
            {"product_id": "p4", "y_true": 4.0, "top1_probability": 0.6, "factor_weighted_mean": 4.1, "text_len": 5},
        ]
    )
    dev_frame = pd.DataFrame(
        [
            {"product_id": "p5", "y_true": 1.5, "top1_probability": 0.85, "factor_weighted_mean": 1.4, "text_len": 2},
            {"product_id": "p6", "y_true": 3.5, "top1_probability": 0.65, "factor_weighted_mean": 3.6, "text_len": 5},
        ]
    )
    artifacts = train_lgbm_quantile_regressor(
        train_frame=train_frame,
        dev_frame=dev_frame,
        output_dir=tmp_path,
        quantiles=(0.1, 0.5, 0.9),
        top_k=2,
        pca_dim=2,
        seed=13,
    )
    assert artifacts.bundle_path.exists()
    assert artifacts.dev_predictions_path.exists()
    assert artifacts.dev_metrics_path.exists()
