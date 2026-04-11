from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from open_match_lca.io_utils import require_exists
from open_match_lca.schemas import validate_non_empty


def _coalesce(record: dict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def parse_uslci_jsonld(uslci_dir: str) -> pd.DataFrame:
    root = require_exists(Path(uslci_dir))
    files = sorted([*root.glob("*.json"), *root.glob("*.jsonld")])
    if not files:
        raise FileNotFoundError(
            f"No USLCI JSON/JSON-LD files found in {root}. "
            "This feature is an optional extension; the main experiment is unaffected."
        )
    rows: list[dict[str, str]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            process_name = _coalesce(record, "process_name", "name")
            category_path = _coalesce(record, "category_path", "category")
            geography = _coalesce(record, "geography", "location")
            flow_name = _coalesce(record, "reference_flow_name", "referenceFlowName")
            flow_unit = _coalesce(record, "reference_flow_unit", "referenceFlowUnit")
            description = _coalesce(record, "description", "comment")
            rows.append(
                {
                    "process_uuid": _coalesce(record, "process_uuid", "@id", "uuid"),
                    "process_name": process_name,
                    "category_path": category_path,
                    "geography": geography,
                    "reference_flow_name": flow_name,
                    "reference_flow_unit": flow_unit,
                    "process_text": " | ".join(
                        part for part in [process_name, category_path, geography, flow_name, description] if part
                    ),
                    "source_release": _coalesce(record, "source_release", "release", "version"),
                }
            )
    frame = pd.DataFrame(rows)
    validate_non_empty(frame, "uslci_processes")
    return frame
