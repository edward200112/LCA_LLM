from __future__ import annotations

import pandas as pd

from open_match_lca.features.hierarchy_features import parent_code, split_naics_levels
from open_match_lca.io_utils import read_tabular_dir
from open_match_lca.schemas import ensure_columns, normalize_naics_code, validate_non_empty

NAICS_INPUT_REQUIRED_COLUMNS = ["naics_code", "naics_title"]


def build_naics_corpus(naics_dir: str) -> pd.DataFrame:
    frame = read_tabular_dir(naics_dir)
    ensure_columns(frame, NAICS_INPUT_REQUIRED_COLUMNS, "naics")
    parsed = frame.copy()
    parsed["naics_code"] = parsed["naics_code"].map(normalize_naics_code)
    parsed["naics_title"] = parsed["naics_title"].fillna("").astype(str)
    if "naics_description" not in parsed.columns:
        parsed["naics_description"] = ""
    parsed["naics_text"] = (parsed["naics_title"] + " " + parsed["naics_description"].fillna("").astype(str)).str.strip()
    levels = parsed["naics_code"].map(split_naics_levels)
    parsed["naics_code_2"] = levels.map(lambda item: item[0])
    parsed["naics_code_4"] = levels.map(lambda item: item[1])
    parsed["naics_code_6"] = levels.map(lambda item: item[2])
    parsed["parent_code"] = parsed["naics_code"].map(parent_code)
    parsed["level"] = parsed["naics_code"].map(lambda code: 6 if code else 0)
    validate_non_empty(parsed, "naics")
    return parsed[
        [
            "naics_code",
            "naics_code_2",
            "naics_code_4",
            "naics_code_6",
            "naics_title",
            "naics_text",
            "parent_code",
            "level",
        ]
    ].drop_duplicates(subset=["naics_code"])
