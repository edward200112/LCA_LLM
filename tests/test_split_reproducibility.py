from __future__ import annotations

import pandas as pd

from open_match_lca.data.make_splits import make_dataset_splits


def _sample_products() -> pd.DataFrame:
    rows = []
    codes = ["111110", "111120", "335139", "337127", "337211", "322230"]
    for index in range(18):
        code = codes[index % len(codes)]
        rows.append(
            {
                "product_id": f"p{index}",
                "text": f"sample product {index} for {code}",
                "gold_naics_code": code,
            }
        )
    return pd.DataFrame(rows)


def test_random_stratified_reproducible() -> None:
    products = _sample_products()
    split_a = make_dataset_splits(products, "random_stratified", 42)
    split_b = make_dataset_splits(products, "random_stratified", 42)
    assert split_a["train"]["product_id"].tolist() == split_b["train"]["product_id"].tolist()
    assert split_a["dev"]["product_id"].tolist() == split_b["dev"]["product_id"].tolist()
    assert split_a["test"]["product_id"].tolist() == split_b["test"]["product_id"].tolist()


def test_cluster_ood_reproducible() -> None:
    products = _sample_products()
    split_a = make_dataset_splits(products, "cluster_ood", 13)
    split_b = make_dataset_splits(products, "cluster_ood", 13)
    assert split_a["test"]["product_id"].tolist() == split_b["test"]["product_id"].tolist()
