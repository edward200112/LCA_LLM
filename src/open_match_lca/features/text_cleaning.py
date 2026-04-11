from __future__ import annotations

import re

from open_match_lca.constants import TEXT_SEPARATOR

_MULTISPACE = re.compile(r"\s+")
_ALLOWED = re.compile(r"[^0-9a-zA-Z%./+\-\sxX]")


def clean_text(text: object) -> str:
    value = "" if text is None else str(text)
    value = _ALLOWED.sub(" ", value)
    value = _MULTISPACE.sub(" ", value).strip()
    return value


def compose_product_text(title: str, description: str) -> str:
    title_clean = clean_text(title)
    description_clean = clean_text(description)
    if not description_clean:
        return title_clean
    return f"{title_clean}{TEXT_SEPARATOR}{description_clean}"


def has_numeric_tokens(text: str) -> bool:
    return any(char.isdigit() for char in text)


def count_numeric_tokens(text: str) -> int:
    return sum(bool(any(char.isdigit() for char in token)) for token in text.split())
