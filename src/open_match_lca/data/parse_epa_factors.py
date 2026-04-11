from __future__ import annotations

import pandas as pd

from open_match_lca.io_utils import read_tabular_dir
from open_match_lca.schemas import ensure_columns, normalize_naics_code, validate_non_empty

EPA_INPUT_REQUIRED_COLUMNS = [
    "naics_code",
    "factor_value",
    "factor_unit",
    "with_margins",
    "without_margins",
    "source_year",
    "useeio_code",
]


def parse_epa_factors(epa_dir: str) -> pd.DataFrame:
    frame = read_tabular_dir(epa_dir)
    ensure_columns(frame, EPA_INPUT_REQUIRED_COLUMNS, "epa_factors")
    parsed = frame.copy()
    parsed["naics_code"] = parsed["naics_code"].map(normalize_naics_code)
    parsed["factor_value"] = parsed["factor_value"].astype(float)
    parsed["source_year"] = parsed["source_year"].astype(int)
    validate_non_empty(parsed, "epa_factors")
    return parsed[EPA_INPUT_REQUIRED_COLUMNS]
