from __future__ import annotations

import pandas as pd


def merge_products_with_epa_factors(
    products: pd.DataFrame,
    epa_factors: pd.DataFrame,
) -> pd.DataFrame:
    merged = products.merge(
        epa_factors,
        how="left",
        left_on="gold_naics_code",
        right_on="naics_code",
        suffixes=("", "_factor"),
    )
    return merged
