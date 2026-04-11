from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

PRODUCT_REQUIRED_COLUMNS = [
    "product_id",
    "title",
    "description",
    "text",
    "gold_naics_code",
    "gold_naics_title",
    "label_confidence",
    "text_len",
    "has_numeric_tokens",
]

EPA_REQUIRED_COLUMNS = [
    "naics_code",
    "factor_value",
    "factor_unit",
    "with_margins",
    "without_margins",
    "source_year",
    "useeio_code",
]

NAICS_REQUIRED_COLUMNS = [
    "naics_code",
    "naics_code_2",
    "naics_code_4",
    "naics_code_6",
    "naics_title",
    "naics_text",
    "parent_code",
    "level",
]

USLCI_REQUIRED_COLUMNS = [
    "process_uuid",
    "process_name",
    "category_path",
    "geography",
    "reference_flow_name",
    "reference_flow_unit",
    "process_text",
    "source_release",
]


@dataclass(frozen=True)
class DatasetSummary:
    sample_count: int
    class_count: int
    missing_rate: dict[str, float]
    duplicate_rate: float


def normalize_naics_code(value: object) -> str:
    if pd.isna(value):
        return ""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return ""
    return digits[:6].zfill(6)


def ensure_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def validate_non_empty(frame: pd.DataFrame, dataset_name: str) -> None:
    if frame.empty:
        raise ValueError(f"{dataset_name} is empty after parsing.")
