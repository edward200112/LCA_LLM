from __future__ import annotations

import pandas as pd

from open_match_lca.retrieval.dense_training import (
    build_dense_training_pairs,
    choose_hard_negative_bucket,
)


def test_choose_hard_negative_bucket_prefers_4_digit_then_2_digit() -> None:
    code4_counts = {"3371": 2, "3351": 1}
    code2_counts = {"33": 3}
    assert choose_hard_negative_bucket("337127", code4_counts, code2_counts) == ("4_digit", "3371")
    assert choose_hard_negative_bucket("335139", code4_counts, code2_counts) == ("2_digit", "33")
    assert choose_hard_negative_bucket("111110", {}, {"11": 1}) == ("global", "global")


def test_build_dense_training_pairs_orders_sibling_codes_into_same_bucket() -> None:
    train_products = pd.DataFrame(
        [
            {"product_id": "p1", "text": "metal chair", "gold_naics_code": "337127"},
            {"product_id": "p2", "text": "wood desk", "gold_naics_code": "337211"},
            {"product_id": "p3", "text": "office stool", "gold_naics_code": "337127"},
            {"product_id": "p4", "text": "task lamp", "gold_naics_code": "335139"},
        ]
    )
    corpus = pd.DataFrame(
        [
            {"naics_code": "337127", "naics_text": "institutional furniture"},
            {"naics_code": "337211", "naics_text": "wood office furniture"},
            {"naics_code": "335139", "naics_text": "electric lamps"},
        ]
    )
    pairs = build_dense_training_pairs(train_products, corpus)
    assert "positive_text" in pairs.columns
    assert pairs.loc[pairs["gold_naics_code"] == "337127", "hard_negative_bucket"].iloc[0] == "33"
    assert pairs.loc[pairs["gold_naics_code"] == "337211", "hard_negative_bucket"].iloc[0] == "33"
    assert pairs["pair_id"].is_unique
