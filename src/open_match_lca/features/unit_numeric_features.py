from __future__ import annotations

from open_match_lca.features.text_cleaning import count_numeric_tokens, has_numeric_tokens


def extract_numeric_features(text: str) -> dict[str, int | bool]:
    return {
        "numeric_token_count": count_numeric_tokens(text),
        "has_numeric_tokens": has_numeric_tokens(text),
    }
