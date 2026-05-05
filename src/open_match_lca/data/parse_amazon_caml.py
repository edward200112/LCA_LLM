from __future__ import annotations

import ast

import pandas as pd

from open_match_lca.features.text_cleaning import compose_product_text, has_numeric_tokens
from open_match_lca.io_utils import read_tabular_input
from open_match_lca.schemas import ensure_columns, normalize_naics_code, validate_non_empty

AMAZON_INPUT_REQUIRED_COLUMNS = ["product_id", "title", "description", "gold_naics_code"]
AMAZON_CAML_ALT_COLUMNS = ["product_code", "product_text", "naics_code"]
AMAZON_INVALID_NAICS_CODES = {"000000", "000001"}
AMAZON_EXTENSION_EXCLUDE_FILENAMES = {"pv_glass_cases.csv"}


def _split_product_text(value: object) -> tuple[str, str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return "", ""
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts:
        return text, ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _parse_label_confidence(raw_annotations: object, gold_naics_code: str) -> float:
    if raw_annotations is None or (isinstance(raw_annotations, float) and pd.isna(raw_annotations)):
        return 1.0
    try:
        values = ast.literal_eval(str(raw_annotations))
    except (SyntaxError, ValueError):
        return 1.0
    if not isinstance(values, (list, tuple)):
        return 1.0
    normalized = [
        normalize_naics_code(value)
        for value in values
        if str(value) not in {"-1", "0", "0.0", "nan", "None"}
    ]
    normalized = [value for value in normalized if value and value != "000000"]
    if not normalized:
        return 1.0
    matches = sum(value == gold_naics_code for value in normalized)
    return float(matches / len(normalized))


def _convert_alt_schema(frame: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(frame, AMAZON_CAML_ALT_COLUMNS, "amazon_caml")
    parsed = frame.copy()
    parsed["product_id"] = parsed["product_code"].astype(str)
    split_values = parsed["product_text"].map(_split_product_text)
    parsed["title"] = split_values.map(lambda item: item[0])
    parsed["description"] = split_values.map(lambda item: item[1])
    parsed["gold_naics_code"] = parsed["naics_code"]
    parsed["label_confidence"] = [
        _parse_label_confidence(raw_annotations, normalize_naics_code(code))
        for raw_annotations, code in zip(
            parsed.get("raw_annotations", pd.Series([None] * len(parsed))),
            parsed["gold_naics_code"],
            strict=False,
        )
    ]
    return parsed


def parse_amazon_caml(amazon_dir: str) -> pd.DataFrame:
    frame = read_tabular_input(amazon_dir, exclude_filenames=AMAZON_EXTENSION_EXCLUDE_FILENAMES)
    if all(column in frame.columns for column in AMAZON_INPUT_REQUIRED_COLUMNS):
        parsed = frame.copy()
    elif all(column in frame.columns for column in AMAZON_CAML_ALT_COLUMNS):
        parsed = _convert_alt_schema(frame)
    else:
        ensure_columns(frame, AMAZON_INPUT_REQUIRED_COLUMNS, "amazon_caml")
        parsed = frame.copy()

    parsed["title"] = parsed["title"].fillna("").astype(str)
    parsed["description"] = parsed["description"].fillna("").astype(str)
    parsed["text"] = [
        compose_product_text(title, description)
        for title, description in zip(parsed["title"], parsed["description"], strict=False)
    ]
    parsed["gold_naics_code"] = parsed["gold_naics_code"].map(normalize_naics_code)
    parsed = parsed.loc[
        ~parsed["gold_naics_code"].isin(AMAZON_INVALID_NAICS_CODES)
        & (parsed["gold_naics_code"].str[:2] != "00")
    ].reset_index(drop=True)
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
