from __future__ import annotations

import pandas as pd

from open_match_lca.features.text_cleaning import compose_product_text, has_numeric_tokens
from open_match_lca.io_utils import read_tabular_dir
from open_match_lca.schemas import ensure_columns, normalize_naics_code, validate_non_empty

AMAZON_INPUT_REQUIRED_COLUMNS = ["product_id", "title", "description", "gold_naics_code"]


def parse_amazon_caml(amazon_dir: str) -> pd.DataFrame:
    frame = read_tabular_dir(amazon_dir)
    ensure_columns(frame, AMAZON_INPUT_REQUIRED_COLUMNS, "amazon_caml")

    parsed = frame.copy()
    parsed["title"] = parsed["title"].fillna("").astype(str)
    parsed["description"] = parsed["description"].fillna("").astype(str)
    parsed["text"] = [
        compose_product_text(title, description)
        for title, description in zip(parsed["title"], parsed["description"], strict=False)
    ]
    parsed["gold_naics_code"] = parsed["gold_naics_code"].map(normalize_naics_code)
    if "gold_naics_title" not in parsed.columns:
        parsed["gold_naics_title"] = ""
    parsed["gold_naics_title"] = parsed["gold_naics_title"].fillna("").astype(str)
    if "label_confidence" not in parsed.columns:
        parsed["label_confidence"] = 1.0
    parsed["label_confidence"] = parsed["label_confidence"].fillna(1.0).astype(float)
    parsed["text_len"] = parsed["text"].str.split().str.len().astype(int)
    parsed["has_numeric_tokens"] = parsed["text"].map(has_numeric_tokens)
    parsed["group_id"] = pd.factorize(parsed["text"])[0].astype(int)
    parsed = parsed.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    validate_non_empty(parsed, "amazon_caml")
    return parsed[
        [
            "product_id",
            "title",
            "description",
            "text",
            "gold_naics_code",
            "gold_naics_title",
            "label_confidence",
            "text_len",
            "has_numeric_tokens",
            "group_id",
        ]
    ]
