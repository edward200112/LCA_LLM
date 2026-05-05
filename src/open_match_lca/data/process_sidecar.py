from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from open_match_lca.constants import PROJECT_ROOT
from open_match_lca.io_utils import read_tabular_path, require_exists, write_parquet
from open_match_lca.schemas import validate_non_empty


PROCESS_COLUMNS = [
    "process_uuid",
    "process_name",
    "category_path",
    "geography",
    "reference_flow_name",
    "reference_flow_unit",
    "process_text",
    "source_release",
    "source_dataset",
    "source_type",
    "source_file",
    "process_kind",
    "naics_code_2",
]

PROCESS_EXCHANGE_COLUMNS = [
    "exchange_id",
    "process_uuid",
    "process_name",
    "exchange_direction",
    "amount",
    "unit",
    "flow_uuid",
    "flow_name",
    "flow_category",
    "flow_type",
    "provider_process_uuid",
    "provider_process_name",
    "is_quantitative_reference",
    "is_avoided_product",
    "exchange_description",
    "source_dataset",
    "source_type",
    "source_file",
    "standardized_item_key",
    "standardized_item_name",
    "is_recommendable",
]

PROCESS_ITEM_COLUMNS = [
    "standardized_item_key",
    "standardized_name",
    "canonical_unit",
    "flow_type",
    "item_text",
    "aliases",
    "source_datasets",
    "supporting_process_count",
    "supporting_exchange_count",
]

PROCESS_CORPUS_COLUMNS = [
    "process_uuid",
    "process_name",
    "category_path",
    "geography",
    "reference_flow_name",
    "reference_flow_unit",
    "process_text",
    "source_release",
    "source_dataset",
    "source_type",
    "source_file",
    "process_kind",
    "naics_code_2",
    "exchange_count",
    "input_exchange_count",
    "top_item_keys",
    "top_item_names",
]

INDEX_REQUIRED_COLUMNS = ["doc_id", "relative_path", "category_level_1", "material_or_product", "file_name"]
AUX_REQUIRED_COLUMNS = [
    "doc_id",
    "title",
    "product_text",
    "category_level_1",
    "material_or_product",
    "manufacturer",
    "source_type",
    "source_file",
    "declared_unit",
    "geography",
]

RECOMMENDABLE_FLOW_TYPES = {"PRODUCT_FLOW", "DOCUMENT_REFERENCE_FLOW"}


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or PROJECT_ROOT).resolve()


def _resolve_repo_path(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def _join_text(parts: list[object]) -> str:
    return re.sub(r"\s+", " ", " | ".join(_clean_text(part) for part in parts if _clean_text(part))).strip()


def _extract_naics_code_2(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    match = re.search(r"\b(\d{2})(?:-\d{2})?:", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{4,6})\b", cleaned)
    if match:
        return match.group(1)[:2]
    return ""


def _standardize_item_name(flow_name: object) -> str:
    text = _clean_text(flow_name).lower()
    if not text:
        return ""
    text = text.replace("_", " ")
    text = re.sub(r"[;,:/]+", " ", text)
    text = re.sub(
        r"\b(at|to|from)\s+(plant|mill|mine|user|treatment plant|facility|site)\b",
        " ",
        text,
    )
    text = re.sub(r"\baverage production\b", " ", text)
    text = re.sub(r"\bunspecified treatment\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|")
    return text


def _standardized_item_key(flow_name: object) -> str:
    standardized_name = _standardize_item_name(flow_name)
    if not standardized_name:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", standardized_name).strip("_")


def _load_index_frame(path: Path, label: str) -> pd.DataFrame:
    index_path = require_exists(path)
    frame = pd.read_csv(index_path, dtype=str, keep_default_na=False).fillna("")
    missing = [column for column in INDEX_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        _warn(f"{label} is missing required columns: {missing}")
        raise ValueError(f"{label} is missing required columns: {missing}")
    return frame


def _load_aux_documents(path: Path) -> pd.DataFrame:
    aux_path = require_exists(path)
    frame = pd.read_csv(aux_path, dtype=str, keep_default_na=False).fillna("")
    missing = [column for column in AUX_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        _warn(f"aux_documents is missing required columns: {missing}")
        raise ValueError(f"aux_documents is missing required columns: {missing}")
    return frame


def _process_text_from_nist_payload(payload: dict[str, Any]) -> str:
    process_doc = payload.get("processDocumentation", {}) or {}
    parts = [
        payload.get("name", ""),
        payload.get("description", ""),
        payload.get("category", ""),
        process_doc.get("technologyDescription", ""),
        process_doc.get("samplingDescription", ""),
        process_doc.get("dataTreatmentDescription", ""),
        process_doc.get("geographyDescription", ""),
        process_doc.get("timeDescription", ""),
        process_doc.get("intendedApplication", ""),
    ]
    return _join_text(parts)


def _reference_exchange(payload: dict[str, Any]) -> tuple[str, str]:
    for exchange in payload.get("exchanges", []) or []:
        if exchange.get("isQuantitativeReference"):
            flow = exchange.get("flow", {}) or {}
            unit = exchange.get("unit", {}) or {}
            return _clean_text(flow.get("name")), _clean_text(unit.get("name"))
    return "", ""


def _parse_nist_source(nist_root: Path, source_release_fallback: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    process_dir = require_exists(nist_root / "processes")
    process_files = sorted(process_dir.glob("*.json"))
    if not process_files:
        _warn(f"No process JSON files found in {process_dir}")
        raise FileNotFoundError(f"No process JSON files found in {process_dir}")

    process_rows: list[dict[str, Any]] = []
    exchange_rows: list[dict[str, Any]] = []
    for path in process_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("@type") != "Process":
            _warn(f"Skipping unsupported NIST record at {path}: expected @type=Process")
            continue

        process_uuid = _clean_text(payload.get("@id") or path.stem)
        if not process_uuid:
            _warn(f"Missing process UUID in {path}")
            raise ValueError(f"Missing process UUID in {path}")
        process_name = _clean_text(payload.get("name") or path.stem)
        category_path = _clean_text(payload.get("category"))
        location = payload.get("location", {}) or {}
        geography = _clean_text(location.get("name"))
        process_text = _process_text_from_nist_payload(payload)
        reference_flow_name, reference_flow_unit = _reference_exchange(payload)

        process_rows.append(
            {
                "process_uuid": process_uuid,
                "process_name": process_name,
                "category_path": category_path,
                "geography": geography,
                "reference_flow_name": reference_flow_name,
                "reference_flow_unit": reference_flow_unit,
                "process_text": process_text,
                "source_release": _clean_text(payload.get("version")) or source_release_fallback,
                "source_dataset": "nist_building_systems",
                "source_type": "openlca_process",
                "source_file": str(path.relative_to(nist_root.parent)),
                "process_kind": _clean_text(payload.get("processType")) or "UNIT_PROCESS",
                "naics_code_2": _extract_naics_code_2(category_path),
            }
        )

        for index, exchange in enumerate(payload.get("exchanges", []) or [], start=1):
            flow = exchange.get("flow", {}) or {}
            unit = exchange.get("unit", {}) or {}
            provider = exchange.get("defaultProvider", {}) or {}
            flow_name = _clean_text(flow.get("name"))
            flow_type = _clean_text(flow.get("flowType"))
            exchange_rows.append(
                {
                    "exchange_id": f"{process_uuid}:{exchange.get('internalId', index)}",
                    "process_uuid": process_uuid,
                    "process_name": process_name,
                    "exchange_direction": "input" if bool(exchange.get("isInput")) else "output",
                    "amount": exchange.get("amount"),
                    "unit": _clean_text(unit.get("name")),
                    "flow_uuid": _clean_text(flow.get("@id")),
                    "flow_name": flow_name,
                    "flow_category": _clean_text(flow.get("category")),
                    "flow_type": flow_type,
                    "provider_process_uuid": _clean_text(provider.get("@id")),
                    "provider_process_name": _clean_text(provider.get("name")),
                    "is_quantitative_reference": bool(exchange.get("isQuantitativeReference")),
                    "is_avoided_product": bool(exchange.get("isAvoidedProduct")),
                    "exchange_description": _clean_text(exchange.get("description")),
                    "source_dataset": "nist_building_systems",
                    "source_type": "openlca_process",
                    "source_file": str(path.relative_to(nist_root.parent)),
                    "standardized_item_key": _standardized_item_key(flow_name),
                    "standardized_item_name": _standardize_item_name(flow_name),
                    "is_recommendable": flow_type in RECOMMENDABLE_FLOW_TYPES,
                }
            )
    return process_rows, exchange_rows


def _aux_source_dataset(category_level_1: str) -> str:
    if category_level_1 == "glass":
        return "glass_epd"
    if category_level_1 == "raw_material":
        return "material_epd"
    return "auxiliary_pdf"


def _parse_auxiliary_sources(
    *,
    glass_index_path: Path,
    material_index_path: Path,
    aux_documents_path: Path,
    source_release_fallback: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    glass_index = _load_index_frame(glass_index_path, "glass_index")
    material_index = _load_index_frame(material_index_path, "material_index")
    aux_documents = _load_aux_documents(aux_documents_path)
    aux_lookup = {
        row["doc_id"]: row
        for row in aux_documents.to_dict(orient="records")
    }

    process_rows: list[dict[str, Any]] = []
    exchange_rows: list[dict[str, Any]] = []
    index_frame = pd.concat([glass_index, material_index], ignore_index=True)
    for row in index_frame.to_dict(orient="records"):
        doc_id = _clean_text(row.get("doc_id"))
        if not doc_id:
            _warn("Encountered auxiliary index row without doc_id")
            raise ValueError("Encountered auxiliary index row without doc_id")
        category_level_1 = _clean_text(row.get("category_level_1"))
        if category_level_1 not in {"glass", "raw_material"}:
            continue
        aux_row = aux_lookup.get(doc_id, {})
        title = _clean_text(aux_row.get("title")) or Path(row.get("file_name", "")).stem
        material_or_product = _clean_text(aux_row.get("material_or_product") or row.get("material_or_product"))
        manufacturer = _clean_text(aux_row.get("manufacturer") or row.get("manufacturer"))
        geography = _clean_text(aux_row.get("geography"))
        declared_unit = _clean_text(aux_row.get("declared_unit"))
        product_text = _clean_text(aux_row.get("product_text"))
        process_name = title or material_or_product or doc_id
        reference_flow_name = material_or_product.replace("_", " ").strip() or process_name
        source_file = _clean_text(aux_row.get("source_file") or row.get("relative_path"))
        source_dataset = _aux_source_dataset(category_level_1)
        naics_code_2 = "32" if category_level_1 == "glass" else ""
        process_rows.append(
            {
                "process_uuid": doc_id,
                "process_name": process_name,
                "category_path": f"auxiliary/{category_level_1}/{material_or_product or 'unknown'}",
                "geography": geography,
                "reference_flow_name": reference_flow_name,
                "reference_flow_unit": declared_unit,
                "process_text": _join_text(
                    [
                        process_name,
                        product_text,
                        category_level_1,
                        material_or_product.replace("_", " "),
                        manufacturer,
                        geography,
                    ]
                ),
                "source_release": source_release_fallback,
                "source_dataset": source_dataset,
                "source_type": _clean_text(aux_row.get("source_type") or row.get("doc_type")) or "pdf",
                "source_file": source_file,
                "process_kind": "document_sidecar",
                "naics_code_2": naics_code_2,
            }
        )
        synthetic_flow_name = reference_flow_name or process_name
        exchange_rows.append(
            {
                "exchange_id": f"{doc_id}:reference",
                "process_uuid": doc_id,
                "process_name": process_name,
                "exchange_direction": "output",
                "amount": 1.0,
                "unit": declared_unit,
                "flow_uuid": "",
                "flow_name": synthetic_flow_name,
                "flow_category": f"document/{category_level_1}",
                "flow_type": "DOCUMENT_REFERENCE_FLOW",
                "provider_process_uuid": "",
                "provider_process_name": "",
                "is_quantitative_reference": True,
                "is_avoided_product": False,
                "exchange_description": _clean_text(aux_row.get("parse_status")),
                "source_dataset": source_dataset,
                "source_type": _clean_text(aux_row.get("source_type") or row.get("doc_type")) or "pdf",
                "source_file": source_file,
                "standardized_item_key": _standardized_item_key(synthetic_flow_name),
                "standardized_item_name": _standardize_item_name(synthetic_flow_name),
                "is_recommendable": True,
            }
        )
    return process_rows, exchange_rows


def _build_process_items_standardized(process_exchanges: pd.DataFrame) -> pd.DataFrame:
    recommendable = process_exchanges.loc[
        process_exchanges["is_recommendable"].fillna(False)
        & process_exchanges["standardized_item_key"].astype(str).ne("")
    ].copy()
    if recommendable.empty:
        _warn("No recommendable process exchanges were found while building standardized items")
        raise ValueError("No recommendable process exchanges were found while building standardized items")

    rows: list[dict[str, Any]] = []
    for item_key, group in recommendable.groupby("standardized_item_key", sort=True):
        aliases = sorted({alias for alias in group["flow_name"].astype(str).tolist() if alias})
        units = [unit for unit in group["unit"].astype(str).tolist() if unit]
        flow_types = [label for label in group["flow_type"].astype(str).tolist() if label]
        source_datasets = sorted({label for label in group["source_dataset"].astype(str).tolist() if label})
        standardized_names = [name for name in group["standardized_item_name"].astype(str).tolist() if name]
        preferred_name = min(standardized_names or aliases, key=lambda value: (len(value), value))
        canonical_unit = pd.Series(units).value_counts().index[0] if units else ""
        flow_type = pd.Series(flow_types).value_counts().index[0] if flow_types else ""
        rows.append(
            {
                "standardized_item_key": item_key,
                "standardized_name": preferred_name,
                "canonical_unit": canonical_unit,
                "flow_type": flow_type,
                "item_text": _join_text([preferred_name, " ".join(aliases[:5]), canonical_unit, flow_type]),
                "aliases": aliases,
                "source_datasets": source_datasets,
                "supporting_process_count": int(group["process_uuid"].nunique()),
                "supporting_exchange_count": int(len(group)),
            }
        )
    frame = pd.DataFrame(rows, columns=PROCESS_ITEM_COLUMNS).sort_values(
        ["supporting_process_count", "standardized_name"],
        ascending=[False, True],
    )
    validate_non_empty(frame, "process_items_standardized")
    return frame.reset_index(drop=True)


def _build_process_corpus(
    processes: pd.DataFrame,
    process_exchanges: pd.DataFrame,
    process_items: pd.DataFrame,
) -> pd.DataFrame:
    item_name_lookup = process_items.set_index("standardized_item_key")["standardized_name"].to_dict()
    recommendable = process_exchanges.loc[
        process_exchanges["standardized_item_key"].astype(str).ne("")
        & process_exchanges["is_recommendable"].fillna(False)
    ].copy()
    recommendable["item_name"] = recommendable["standardized_item_key"].map(item_name_lookup).fillna("")
    grouped_items: dict[str, list[str]] = {}
    grouped_keys: dict[str, list[str]] = {}
    for process_uuid, group in recommendable.groupby("process_uuid", sort=False):
        working = group.copy()
        working["priority"] = 0
        working.loc[working["exchange_direction"].eq("input"), "priority"] = 2
        working.loc[working["is_quantitative_reference"].fillna(False), "priority"] = 1
        working = working.sort_values(["priority", "flow_name"], ascending=[False, True])
        item_keys = []
        item_names = []
        for row in working.to_dict(orient="records"):
            item_key = _clean_text(row.get("standardized_item_key"))
            item_name = _clean_text(row.get("item_name"))
            if not item_key:
                continue
            if item_key not in item_keys:
                item_keys.append(item_key)
            if item_name and item_name not in item_names:
                item_names.append(item_name)
            if len(item_keys) >= 12 and len(item_names) >= 12:
                break
        grouped_keys[process_uuid] = item_keys
        grouped_items[process_uuid] = item_names

    exchange_counts = process_exchanges.groupby("process_uuid").size().to_dict()
    input_counts = process_exchanges.loc[process_exchanges["exchange_direction"].eq("input")].groupby("process_uuid").size().to_dict()

    rows: list[dict[str, Any]] = []
    for row in processes.to_dict(orient="records"):
        item_names = grouped_items.get(row["process_uuid"], [])
        item_keys = grouped_keys.get(row["process_uuid"], [])
        retrieval_text = _join_text(
            [
                row["process_name"],
                row["category_path"],
                row["geography"],
                row["reference_flow_name"],
                row["process_text"],
                "Inputs " + " ".join(item_names[:12]) if item_names else "",
            ]
        )
        rows.append(
            {
                **row,
                "process_text": retrieval_text,
                "exchange_count": int(exchange_counts.get(row["process_uuid"], 0)),
                "input_exchange_count": int(input_counts.get(row["process_uuid"], 0)),
                "top_item_keys": item_keys,
                "top_item_names": item_names,
            }
        )
    frame = pd.DataFrame(rows, columns=PROCESS_CORPUS_COLUMNS).sort_values(
        ["source_dataset", "process_name"],
        ascending=[True, True],
    )
    validate_non_empty(frame, "process_corpus")
    return frame.reset_index(drop=True)


def prepare_process_sidecar(
    config: dict[str, Any],
    repo_root: Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Path]:
    root = _repo_root(repo_root)
    outputs = {
        "processes": _resolve_repo_path(root, config.get("processes_output", "data/processed/processes.parquet")),
        "process_exchanges": _resolve_repo_path(
            root,
            config.get("process_exchanges_output", "data/processed/process_exchanges.parquet"),
        ),
        "process_items": _resolve_repo_path(
            root,
            config.get("process_items_output", "data/processed/process_items_standardized.parquet"),
        ),
        "process_corpus": _resolve_repo_path(root, config.get("process_corpus_output", "data/processed/process_corpus.parquet")),
    }
    if not force and all(path.exists() for path in outputs.values()):
        return outputs

    source_release_fallback = _clean_text(config.get("source_release_fallback")) or "process_sidecar_v1"
    nist_root = _resolve_repo_path(root, config.get("nist_root", "data/NIST-Building_Systems"))
    process_rows, exchange_rows = _parse_nist_source(nist_root, source_release_fallback)

    if bool(config.get("include_auxiliary_pdf_processes", True)):
        aux_process_rows, aux_exchange_rows = _parse_auxiliary_sources(
            glass_index_path=_resolve_repo_path(root, config.get("glass_index_path", "data/Glass_EPD/index.csv")),
            material_index_path=_resolve_repo_path(root, config.get("material_index_path", "data/Material_EPD/index.csv")),
            aux_documents_path=_resolve_repo_path(root, config.get("aux_documents_path", "data/interim/aux_documents.csv")),
            source_release_fallback=source_release_fallback,
        )
        process_rows.extend(aux_process_rows)
        exchange_rows.extend(aux_exchange_rows)

    processes = pd.DataFrame(process_rows, columns=PROCESS_COLUMNS)
    validate_non_empty(processes, "processes")
    if processes["process_uuid"].duplicated().any():
        duplicated = sorted(processes.loc[processes["process_uuid"].duplicated(), "process_uuid"].unique().tolist())
        _warn(f"Duplicate process UUIDs detected: {duplicated[:10]}")
        raise ValueError(f"Duplicate process UUIDs detected: {duplicated[:10]}")
    processes = processes.sort_values(["source_dataset", "process_name"]).reset_index(drop=True)

    process_exchanges = pd.DataFrame(exchange_rows, columns=PROCESS_EXCHANGE_COLUMNS)
    validate_non_empty(process_exchanges, "process_exchanges")
    if process_exchanges["exchange_id"].duplicated().any():
        duplicated = sorted(process_exchanges.loc[process_exchanges["exchange_id"].duplicated(), "exchange_id"].unique().tolist())
        _warn(f"Duplicate exchange IDs detected: {duplicated[:10]}")
        raise ValueError(f"Duplicate exchange IDs detected: {duplicated[:10]}")
    process_exchanges["amount"] = pd.to_numeric(process_exchanges["amount"], errors="coerce")
    process_exchanges = process_exchanges.sort_values(["process_uuid", "exchange_id"]).reset_index(drop=True)

    process_items = _build_process_items_standardized(process_exchanges)
    process_corpus = _build_process_corpus(processes, process_exchanges, process_items)

    write_parquet(processes, outputs["processes"])
    write_parquet(process_exchanges, outputs["process_exchanges"])
    write_parquet(process_items, outputs["process_items"])
    write_parquet(process_corpus, outputs["process_corpus"])
    return outputs


def load_process_sidecar_table(path: str | Path) -> pd.DataFrame:
    return read_tabular_path(require_exists(Path(path)))
