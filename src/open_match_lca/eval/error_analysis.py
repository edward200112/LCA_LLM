from __future__ import annotations

from pathlib import Path

from open_match_lca.io_utils import write_jsonl


def export_error_cases(records: list[dict], path: str | Path) -> None:
    errors = []
    for record in records:
        candidates = record.get("candidates", [])
        pred = candidates[0].get("naics_code") if candidates else None
        if pred != record.get("gold_naics_code"):
            errors.append(record)
    write_jsonl(errors, path)
