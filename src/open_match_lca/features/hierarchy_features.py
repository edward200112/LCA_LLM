from __future__ import annotations

from open_match_lca.schemas import normalize_naics_code


def split_naics_levels(naics_code: str) -> tuple[str, str, str]:
    code = normalize_naics_code(naics_code)
    return code[:2], code[:4], code[:6]


def parent_code(naics_code: str) -> str:
    code = normalize_naics_code(naics_code)
    return code[:4] if len(code) >= 4 else ""


def hierarchical_distance(left: str, right: str) -> int:
    left_code = normalize_naics_code(left)
    right_code = normalize_naics_code(right)
    if left_code == right_code:
        return 0
    if left_code[:4] == right_code[:4]:
        return 1
    if left_code[:2] == right_code[:2]:
        return 2
    return 3
