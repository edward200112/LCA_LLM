from __future__ import annotations

import pandas as pd

from open_match_lca.io_utils import dump_json
from open_match_lca.schemas import DatasetSummary


def summarize_products(products: pd.DataFrame) -> DatasetSummary:
    missing_rate = {
        column: float(products[column].isna().mean())
        for column in products.columns
    }
    duplicate_rate = float(products["text"].duplicated().mean()) if "text" in products else 0.0
    return DatasetSummary(
        sample_count=int(len(products)),
        class_count=int(products["gold_naics_code"].nunique()) if "gold_naics_code" in products else 0,
        missing_rate=missing_rate,
        duplicate_rate=duplicate_rate,
    )


def write_summary_report(products: pd.DataFrame, path: str) -> dict:
    summary = summarize_products(products)
    payload = {
        "sample_count": summary.sample_count,
        "class_count": summary.class_count,
        "missing_rate": summary.missing_rate,
        "duplicate_rate": summary.duplicate_rate,
        "class_distribution_top10": products["gold_naics_code"].value_counts().head(10).to_dict(),
    }
    dump_json(payload, path)
    return payload
