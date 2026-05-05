from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from open_match_lca.constants import PROJECT_ROOT
from open_match_lca.data.aux_pdf_parser import clean_filename_title, extract_thickness_mm
from open_match_lca.features.text_cleaning import clean_text
from open_match_lca.io_utils import dump_json, ensure_directory, read_json, require_exists, write_parquet
from open_match_lca.retrieval.candidate_generation import lexical_overlap_score


OPENLCA_HYBRID_GROUPS: dict[str, str] = {
    "USLCI": "data/USLCI",
    "US Electricity Baseline": "data/US Electricity Baseline",
    "TRACI": "data/TRACI",
    "bridge processes": "data/bridge processes",
    "Federal_LCA_Commons": "data/Federal_LCA_Commons",
    "USEEIO repository": "data/USEEIO repository",
    "Glass_EPD": "data/Glass_EPD",
    "Material_EPD": "data/Material_EPD",
    "glass_baseline": "data/glass_baseline",
    "interim": "data/interim",
    "processed": "data/processed",
}

AUDIT_COLUMNS = [
    "asset_path",
    "asset_name",
    "asset_group",
    "file_type",
    "size_bytes",
    "inferred_role",
    "likely_importable_to_openlca",
    "likely_registry_source",
    "likely_reference_source",
    "notes",
]

PROCESS_REGISTRY_COLUMNS = [
    "process_uuid",
    "process_name",
    "process_category",
    "location",
    "reference_flow_name",
    "reference_flow_unit",
    "source_repo",
    "source_version",
    "has_product_system",
    "calculable_flag",
    "calc_blocker_reason",
    "retrieval_text",
    "asset_path",
    "process_text",
    "process_type",
]

METHOD_REGISTRY_COLUMNS = [
    "method_name",
    "method_uuid",
    "source_repo",
    "asset_path",
    "method_family",
    "notes",
]

REFERENCE_REGISTRY_COLUMNS = [
    "reference_id",
    "source_type",
    "source_name",
    "product_scope",
    "geography",
    "system_boundary",
    "declared_unit",
    "factor_value",
    "factor_unit",
    "thickness_mm",
    "density_kg_m3",
    "mass_per_m2",
    "asset_path",
    "note",
]


@dataclass(frozen=True)
class NormalizationResult:
    normalized_value: float | None
    normalized_unit: str
    result_status: str
    note: str
    mass_per_m2: float | None


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or PROJECT_ROOT).resolve()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except Exception:
        return str(path.resolve())


def _safe_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return 0
    except Exception:
        return 0


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def _clean_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _clean_optional_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value).lower()
    return text in {"1", "true", "yes", "y"}


def _join_text(parts: list[object]) -> str:
    return " | ".join(part for part in (_clean_text(part) for part in parts) if part)


def _normalize_unit(value: object) -> str:
    text = _clean_text(value).lower()
    text = text.replace("square meter", "m2")
    text = text.replace("square metre", "m2")
    text = text.replace("m^2", "m2")
    text = text.replace("kilogram", "kg")
    text = text.replace("kg co2 eq", "kgco2e")
    text = text.replace("kg co2-eq", "kgco2e")
    text = text.replace("kg co2 equiv.", "kgco2e")
    text = text.replace("kg co2 equivalent", "kgco2e")
    text = text.replace("kg co2-e", "kgco2e")
    text = text.replace("kg co2", "kgco2e")
    text = text.replace(" ", "")
    return text


def _looks_like_kg_unit(unit: object) -> bool:
    return _normalize_unit(unit) in {"kg", "kilograms", "kilogram"}


def _looks_like_m2_unit(unit: object) -> bool:
    return _normalize_unit(unit) in {"m2", "sqm", "m²"}


def _is_openlca_export_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_manifest = (path / "openlca.json").exists() or (path / "olca-schema.json").exists()
    if not has_manifest:
        return False
    expected = {"processes", "flows", "lcia_methods", "locations", "sources", "unit_groups"}
    present = {child.name for child in path.iterdir() if child.is_dir()}
    return bool(expected & present)


def _dir_openlca_profile(path: Path) -> str:
    names = {child.name for child in path.iterdir() if child.is_dir()}
    if "processes" in names:
        return "openlca_process_directory"
    if "lcia_methods" in names:
        return "openlca_method_directory"
    if "flows" in names:
        return "openlca_flow_directory"
    return "openlca_export_directory"


def _zip_profile(path: Path) -> tuple[str, str]:
    try:
        with ZipFile(path) as handle:
            names = set(handle.namelist())
    except Exception as exc:
        return path.suffix.lower().lstrip(".") or "zip", f"zip_unreadable: {exc}"
    if "xl/workbook.xml" in names:
        return "xlsx", "excel_workbook_archive"
    if "openlca.json" in names or any(name.endswith("/openlca.json") for name in names):
        prefixes = []
        for token in ("processes/", "flows/", "lcia_methods/", "locations/"):
            if any(token in name for name in names):
                prefixes.append(token.rstrip("/"))
        detail = ", ".join(prefixes) if prefixes else "openlca_layout"
        return "openlca_export_zip", detail
    return "zip", "generic_archive"


def _infer_role(asset_group: str, path: Path, file_type: str) -> str:
    if asset_group == "USLCI":
        return "uslci_source"
    if asset_group == "US Electricity Baseline":
        return "electricity_baseline_source"
    if asset_group == "TRACI":
        return "traci_method_source"
    if asset_group == "bridge processes":
        return "bridge_process_source"
    if asset_group == "Federal_LCA_Commons":
        return "flcac_metadata_source"
    if asset_group == "USEEIO repository":
        return "useeio_process_source"
    if asset_group in {"Glass_EPD", "Material_EPD", "glass_baseline"}:
        return "pv_glass_reference_source"
    name = path.name.lower()
    if "pv_glass" in name or "glass_factor" in name:
        return "pv_glass_reference_source"
    if "openlca" in name and file_type in {"parquet", "csv", "json"}:
        return "bridge_process_source"
    return "unknown"


def _importable_to_openlca(file_type: str) -> bool:
    return file_type in {
        "zolca",
        "openlca_export_zip",
        "openlca_export_directory",
        "openlca_process_directory",
        "openlca_method_directory",
        "openlca_flow_directory",
    }


def _likely_registry_source(role: str) -> bool:
    return role in {
        "uslci_source",
        "traci_method_source",
        "bridge_process_source",
        "useeio_process_source",
    }


def _likely_reference_source(role: str) -> bool:
    return role == "pv_glass_reference_source"


def _asset_note(path: Path, file_type: str) -> str:
    if file_type.startswith("openlca_") and path.is_dir():
        children = sorted(child.name for child in path.iterdir() if child.is_dir())
        return f"openlca_subdirs={','.join(children[:12])}"
    if file_type == "xlsx":
        try:
            return f"sheets={','.join(read_xlsx_sheet_names(path)[:12])}"
        except Exception as exc:
            return f"xlsx_unreadable: {exc}"
    if path.name.endswith(".zip"):
        _, detail = _zip_profile(path)
        return detail
    if path.is_dir() and not any(path.iterdir()):
        return "empty_directory"
    return ""


def audit_openlca_local_assets(
    repo_root: Path | None = None,
    groups: dict[str, str] | None = None,
) -> pd.DataFrame:
    root = _repo_root(repo_root)
    audit_groups = groups or OPENLCA_HYBRID_GROUPS
    rows: list[dict[str, Any]] = []

    for asset_group, relative_root in audit_groups.items():
        group_root = root / relative_root
        if not group_root.exists():
            rows.append(
                {
                    "asset_path": relative_root,
                    "asset_name": Path(relative_root).name,
                    "asset_group": asset_group,
                    "file_type": "missing",
                    "size_bytes": 0,
                    "inferred_role": _infer_role(asset_group, group_root, "missing"),
                    "likely_importable_to_openlca": False,
                    "likely_registry_source": _likely_registry_source(_infer_role(asset_group, group_root, "missing")),
                    "likely_reference_source": _likely_reference_source(_infer_role(asset_group, group_root, "missing")),
                    "notes": "configured_group_missing",
                }
            )
            continue

        rows.append(
            {
                "asset_path": _relative_path(group_root, root),
                "asset_name": group_root.name,
                "asset_group": asset_group,
                "file_type": _dir_openlca_profile(group_root) if _is_openlca_export_dir(group_root) else "directory",
                "size_bytes": 0,
                "inferred_role": _infer_role(
                    asset_group,
                    group_root,
                    _dir_openlca_profile(group_root) if _is_openlca_export_dir(group_root) else "directory",
                ),
                "likely_importable_to_openlca": _importable_to_openlca(
                    _dir_openlca_profile(group_root) if _is_openlca_export_dir(group_root) else "directory"
                ),
                "likely_registry_source": _likely_registry_source(_infer_role(asset_group, group_root, "directory")),
                "likely_reference_source": _likely_reference_source(_infer_role(asset_group, group_root, "directory")),
                "notes": _asset_note(group_root, "directory"),
            }
        )

        for current_root, dir_names, file_names in os.walk(group_root):
            current = Path(current_root)
            if current != group_root and _is_openlca_export_dir(current):
                file_type = _dir_openlca_profile(current)
                role = _infer_role(asset_group, current, file_type)
                rows.append(
                    {
                        "asset_path": _relative_path(current, root),
                        "asset_name": current.name,
                        "asset_group": asset_group,
                        "file_type": file_type,
                        "size_bytes": 0,
                        "inferred_role": role,
                        "likely_importable_to_openlca": _importable_to_openlca(file_type),
                        "likely_registry_source": _likely_registry_source(role),
                        "likely_reference_source": _likely_reference_source(role),
                        "notes": _asset_note(current, file_type),
                    }
                )

            for dir_name in sorted(dir_names):
                dir_path = current / dir_name
                if _is_openlca_export_dir(dir_path):
                    continue
                if current == group_root:
                    role = _infer_role(asset_group, dir_path, "directory")
                    rows.append(
                        {
                            "asset_path": _relative_path(dir_path, root),
                            "asset_name": dir_path.name,
                            "asset_group": asset_group,
                            "file_type": "directory",
                            "size_bytes": 0,
                            "inferred_role": role,
                            "likely_importable_to_openlca": False,
                            "likely_registry_source": _likely_registry_source(role),
                            "likely_reference_source": _likely_reference_source(role),
                            "notes": _asset_note(dir_path, "directory"),
                        }
                    )

            for file_name in sorted(file_names):
                file_path = current / file_name
                suffix = file_path.suffix.lower()
                file_type = suffix.lstrip(".") or "no_extension"
                notes = ""
                if suffix == ".zip":
                    file_type, detail = _zip_profile(file_path)
                    notes = detail
                role = _infer_role(asset_group, file_path, file_type)
                rows.append(
                    {
                        "asset_path": _relative_path(file_path, root),
                        "asset_name": file_path.name,
                        "asset_group": asset_group,
                        "file_type": file_type,
                        "size_bytes": _safe_size(file_path),
                        "inferred_role": role,
                        "likely_importable_to_openlca": _importable_to_openlca(file_type),
                        "likely_registry_source": _likely_registry_source(role),
                        "likely_reference_source": _likely_reference_source(role),
                        "notes": notes or _asset_note(file_path, file_type),
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    frame = frame.drop_duplicates(subset=["asset_path", "file_type"]).reset_index(drop=True)
    for column in AUDIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[AUDIT_COLUMNS]


def _audit_summary_lines(audit_frame: pd.DataFrame) -> list[str]:
    lines = ["# openLCA Local Asset Audit", ""]
    if audit_frame.empty:
        lines.append("No assets found.")
        return lines

    lines.extend(["## Group Summary", ""])
    group_summary = (
        audit_frame.groupby(["asset_group", "inferred_role"], dropna=False)
        .size()
        .reset_index(name="asset_count")
        .sort_values(["asset_group", "asset_count"], ascending=[True, False])
    )
    lines.append("| asset_group | inferred_role | asset_count |")
    lines.append("|---|---:|---:|")
    for row in group_summary.itertuples(index=False):
        lines.append(f"| {row.asset_group} | {row.inferred_role} | {row.asset_count} |")

    lines.extend(["", "## Key Importable Assets", ""])
    importable = audit_frame.loc[audit_frame["likely_importable_to_openlca"] == True].copy()
    if importable.empty:
        lines.append("No obviously importable openLCA assets were detected.")
    else:
        lines.append("| asset_path | file_type | inferred_role |")
        lines.append("|---|---|---|")
        for row in importable.sort_values("asset_path").itertuples(index=False):
            lines.append(f"| {row.asset_path} | {row.file_type} | {row.inferred_role} |")

    lines.extend(["", "## Key Reference Assets", ""])
    references = audit_frame.loc[audit_frame["likely_reference_source"] == True].copy()
    if references.empty:
        lines.append("No reference-oriented assets were detected.")
    else:
        lines.append("| asset_group | asset_path | file_type |")
        lines.append("|---|---|---|")
        for row in references.sort_values("asset_path").head(40).itertuples(index=False):
            lines.append(f"| {row.asset_group} | {row.asset_path} | {row.file_type} |")
        if len(references) > 40:
            lines.append("")
            lines.append(f"... truncated {len(references) - 40} additional reference rows in CSV/JSON audit outputs.")

    return lines


def write_openlca_audit_outputs(
    audit_frame: pd.DataFrame,
    json_path: str | Path,
    csv_path: str | Path,
    md_path: str | Path,
) -> None:
    json_out = Path(json_path)
    csv_out = Path(csv_path)
    md_out = Path(md_path)
    ensure_directory(json_out.parent)
    ensure_directory(csv_out.parent)
    ensure_directory(md_out.parent)

    payload = {
        "records": audit_frame.to_dict(orient="records"),
        "summary": {
            "asset_groups": audit_frame["asset_group"].value_counts().to_dict() if not audit_frame.empty else {},
            "file_types": audit_frame["file_type"].value_counts().to_dict() if not audit_frame.empty else {},
            "roles": audit_frame["inferred_role"].value_counts().to_dict() if not audit_frame.empty else {},
        },
    }
    dump_json(payload, json_out)
    audit_frame.to_csv(csv_out, index=False)
    md_out.write_text("\n".join(_audit_summary_lines(audit_frame)), encoding="utf-8")


def load_openlca_audit_records(path: str | Path) -> pd.DataFrame:
    payload = read_json(path)
    records = payload.get("records", [])
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    for column in AUDIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[AUDIT_COLUMNS]


def summarize_audit_for_terminal(audit_frame: pd.DataFrame) -> str:
    if audit_frame.empty:
        return "No local openLCA-related assets detected."
    importable = int(audit_frame["likely_importable_to_openlca"].fillna(False).sum())
    registry_sources = int(audit_frame["likely_registry_source"].fillna(False).sum())
    reference_sources = int(audit_frame["likely_reference_source"].fillna(False).sum())
    counts = audit_frame["inferred_role"].value_counts().to_dict()
    top_roles = ", ".join(f"{role}={count}" for role, count in counts.items())
    return (
        f"audited_assets={len(audit_frame)}; importable_openlca_assets={importable}; "
        f"registry_like_assets={registry_sources}; reference_like_assets={reference_sources}; roles: {top_roles}"
    )


def read_xlsx_sheet_names(path: str | Path) -> list[str]:
    workbook_path = require_exists(Path(path))
    with ZipFile(workbook_path) as handle:
        root = ET.fromstring(handle.read("xl/workbook.xml"))
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sheets = root.find("a:sheets", namespace)
    if sheets is None:
        return []
    return [sheet.attrib.get("name", "") for sheet in sheets]


def _xlsx_rel_map(handle: ZipFile) -> dict[str, str]:
    root = ET.fromstring(handle.read("xl/_rels/workbook.xml.rels"))
    return {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in root
        if rel.attrib.get("Target")
    }


def _xlsx_shared_strings(handle: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in handle.namelist():
        return []
    root = ET.fromstring(handle.read("xl/sharedStrings.xml"))
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    values: list[str] = []
    for item in root.findall(f"{namespace}si"):
        values.append("".join(text.text or "" for text in item.iter(f"{namespace}t")))
    return values


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        text = "".join(node.text or "" for node in cell.iter(f"{namespace}t"))
        return _clean_text(text)
    value_node = cell.find(f"{namespace}v")
    if value_node is None:
        return ""
    raw_value = value_node.text or ""
    if cell_type == "s":
        try:
            return _clean_text(shared_strings[int(raw_value)])
        except Exception:
            return _clean_text(raw_value)
    return _clean_text(raw_value)


def read_xlsx_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    workbook_path = require_exists(Path(path))
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    with ZipFile(workbook_path) as handle:
        workbook = ET.fromstring(handle.read("xl/workbook.xml"))
        rel_map = _xlsx_rel_map(handle)
        shared_strings = _xlsx_shared_strings(handle)
        sheets = workbook.find("a:sheets", namespace)
        if sheets is None:
            return pd.DataFrame()
        target = None
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
                rel_target = rel_map.get(rel_id, "")
                if rel_target:
                    target = "xl/" + rel_target.lstrip("/")
                break
        if not target:
            return pd.DataFrame()
        sheet_root = ET.fromstring(handle.read(target))

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows: list[list[str]] = []
    for row in sheet_root.iter(f"{ns}row"):
        rows.append([_xlsx_cell_value(cell, shared_strings) for cell in row.findall(f"{ns}c")])

    if not rows:
        return pd.DataFrame()
    header = rows[0]
    width = len(header)
    normalized_rows = []
    for raw_row in rows[1:]:
        row_values = list(raw_row[:width])
        while len(row_values) < width:
            row_values.append("")
        normalized_rows.append(row_values)
    clean_header = [column or f"column_{index}" for index, column in enumerate(header, start=1)]
    return pd.DataFrame(normalized_rows, columns=clean_header)


def _parse_location_name(payload: dict[str, Any]) -> str:
    location = payload.get("location")
    if isinstance(location, dict):
        return _clean_text(location.get("name") or location.get("category"))
    return _clean_text(location)


def _reference_exchange(payload: dict[str, Any]) -> tuple[str, str]:
    for exchange in payload.get("exchanges", []) or []:
        if exchange.get("isQuantitativeReference"):
            flow = exchange.get("flow", {}) or {}
            unit = exchange.get("unit", {}) or {}
            return _clean_text(flow.get("name")), _clean_text(unit.get("name") or flow.get("refUnit"))
    return "", ""


def _product_system_presence(repo_dir: Path) -> bool:
    for candidate in (repo_dir / "product_systems", repo_dir / "bin" / "product_systems"):
        if candidate.exists() and any(candidate.glob("*.json")):
            return True
    return False


def _parse_openlca_process_directory(repo_dir: Path, source_repo: str) -> pd.DataFrame:
    process_dir = require_exists(repo_dir / "processes")
    has_product_system = _product_system_presence(repo_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(process_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("@type") != "Process":
            continue
        process_uuid = _clean_text(payload.get("@id") or path.stem)
        process_name = _clean_text(payload.get("name") or path.stem)
        process_category = _clean_text(payload.get("category"))
        location = _parse_location_name(payload)
        reference_flow_name, reference_flow_unit = _reference_exchange(payload)
        process_type = _clean_text(payload.get("processType")) or "UNKNOWN"
        source_version = _clean_text(payload.get("version"))

        blockers = []
        if not process_uuid:
            blockers.append("missing_process_uuid")
        if not reference_flow_name:
            blockers.append("missing_reference_flow_name")
        if not reference_flow_unit:
            blockers.append("missing_reference_flow_unit")
        if process_type not in {"UNIT_PROCESS", "LCI_RESULT"}:
            blockers.append(f"unsupported_process_type:{process_type or 'unknown'}")

        retrieval_text = _join_text(
            [
                process_name,
                process_category,
                location,
                reference_flow_name,
                source_repo,
                payload.get("description"),
                payload.get("processDocumentation", {}).get("technologyDescription"),
                payload.get("processDocumentation", {}).get("geographyDescription"),
            ]
        )
        rows.append(
            {
                "process_uuid": process_uuid,
                "process_name": process_name,
                "process_category": process_category,
                "location": location,
                "reference_flow_name": reference_flow_name,
                "reference_flow_unit": reference_flow_unit,
                "source_repo": source_repo,
                "source_version": source_version,
                "has_product_system": has_product_system,
                "calculable_flag": not blockers,
                "calc_blocker_reason": ";".join(blockers),
                "retrieval_text": retrieval_text,
                "asset_path": _relative_path(path, _repo_root()),
                "process_text": retrieval_text,
                "process_type": process_type,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=PROCESS_REGISTRY_COLUMNS)
    return frame[PROCESS_REGISTRY_COLUMNS]


def _parse_traci_method_directory(repo_dir: Path, source_repo: str) -> pd.DataFrame:
    method_dir = require_exists(repo_dir / "lcia_methods")
    rows: list[dict[str, Any]] = []
    for path in sorted(method_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("@type") != "ImpactMethod":
            continue
        impact_categories = payload.get("impactCategories") or []
        rows.append(
            {
                "method_name": _clean_text(payload.get("name") or path.stem),
                "method_uuid": _clean_text(payload.get("@id") or path.stem),
                "source_repo": source_repo,
                "asset_path": _relative_path(path, _repo_root()),
                "method_family": _clean_text(payload.get("category") or payload.get("name")),
                "notes": f"impact_category_count={len(impact_categories)}; version={_clean_text(payload.get('version'))}",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=METHOD_REGISTRY_COLUMNS)
    return frame[METHOD_REGISTRY_COLUMNS]


def _parse_useeio_metadata_workbook(path: Path) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    for sheet_name in ("commodities_meta", "SectorCrosswalk"):
        frame = read_xlsx_sheet(path, sheet_name)
        if frame.empty:
            continue
        frame = frame.fillna("")
        if {"Name", "Code"}.issubset(frame.columns):
            for row in frame.itertuples(index=False):
                process_name = _clean_text(getattr(row, "Name", ""))
                code = _clean_text(getattr(row, "Code", ""))
                location = _clean_text(getattr(row, "Location", ""))
                category = _clean_text(getattr(row, "Category", ""))
                description = _clean_text(getattr(row, "Description", ""))
                reference_unit = _clean_text(getattr(row, "Unit", ""))
                candidates.append(
                    pd.DataFrame(
                        [
                            {
                                "process_uuid": "",
                                "process_name": process_name,
                                "process_category": category or code,
                                "location": location,
                                "reference_flow_name": process_name,
                                "reference_flow_unit": reference_unit,
                                "source_repo": "useeio_repository",
                                "source_version": path.stem,
                                "has_product_system": False,
                                "calculable_flag": False,
                                "calc_blocker_reason": "metadata_only_workbook_no_openlca_process_uuid",
                                "retrieval_text": _join_text(
                                    [process_name, code, category, location, description, "useeio_repository"]
                                ),
                                "asset_path": _relative_path(path, _repo_root()),
                                "process_text": _join_text(
                                    [process_name, code, category, location, description, "useeio_repository"]
                                ),
                                "process_type": "METADATA_ONLY",
                            }
                        ]
                    )
                )
    if not candidates:
        return pd.DataFrame(columns=PROCESS_REGISTRY_COLUMNS)
    frame = pd.concat(candidates, ignore_index=True).drop_duplicates(
        subset=["process_name", "process_category", "location", "reference_flow_unit"]
    )
    return frame[PROCESS_REGISTRY_COLUMNS]


def _parse_standardized_process_corpus(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PROCESS_REGISTRY_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        asset_path = _clean_text(getattr(row, "source_file", ""))
        process_uuid = _clean_text(getattr(row, "process_uuid", ""))
        reference_flow_name = _clean_text(getattr(row, "reference_flow_name", ""))
        reference_flow_unit = _clean_text(getattr(row, "reference_flow_unit", ""))
        blockers = []
        if not process_uuid:
            blockers.append("missing_process_uuid")
        if not reference_flow_name:
            blockers.append("missing_reference_flow_name")
        if not reference_flow_unit:
            blockers.append("missing_reference_flow_unit")
        if asset_path and not (_repo_root() / asset_path).exists():
            blockers.append("missing_backing_asset")

        retrieval_text = _join_text(
            [
                getattr(row, "process_name", ""),
                getattr(row, "category_path", ""),
                getattr(row, "geography", ""),
                reference_flow_name,
                getattr(row, "source_type", ""),
                getattr(row, "process_text", ""),
            ]
        )
        rows.append(
            {
                "process_uuid": process_uuid,
                "process_name": _clean_text(getattr(row, "process_name", "")),
                "process_category": _clean_text(getattr(row, "category_path", "")),
                "location": _clean_text(getattr(row, "geography", "")),
                "reference_flow_name": reference_flow_name,
                "reference_flow_unit": reference_flow_unit,
                "source_repo": _clean_text(getattr(row, "source_type", "")) or "pv_glass_process_corpus_standardized",
                "source_version": _clean_text(getattr(row, "source_release", "")),
                "has_product_system": False,
                "calculable_flag": not blockers,
                "calc_blocker_reason": ";".join(blockers),
                "retrieval_text": retrieval_text,
                "asset_path": asset_path,
                "process_text": retrieval_text,
                "process_type": "OPENLCA_CORPUS_REFERENCE",
            }
        )
    if not rows:
        return pd.DataFrame(columns=PROCESS_REGISTRY_COLUMNS)
    return pd.DataFrame(rows)[PROCESS_REGISTRY_COLUMNS]


def build_openlca_hybrid_registry(
    audit_frame: pd.DataFrame,
    repo_root: Path | None = None,
    standardized_process_corpus_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = _repo_root(repo_root)
    process_frames: list[pd.DataFrame] = []
    method_frames: list[pd.DataFrame] = []
    repo_rows: list[dict[str, Any]] = []

    importable = audit_frame.loc[audit_frame["likely_importable_to_openlca"] == True].copy()
    for row in importable.itertuples(index=False):
        asset_path = root / str(row.asset_path)
        if row.inferred_role == "uslci_source" and asset_path.is_dir():
            parsed = _parse_openlca_process_directory(asset_path, "uslci")
            process_frames.append(parsed)
            repo_rows.append(
                {
                    "source_repo": "uslci",
                    "asset_path": row.asset_path,
                    "asset_group": row.asset_group,
                    "file_type": row.file_type,
                    "process_rows": len(parsed),
                    "method_rows": 0,
                    "calculable_rows": int(parsed["calculable_flag"].fillna(False).sum()) if not parsed.empty else 0,
                    "notes": row.notes,
                }
            )
        elif row.inferred_role == "traci_method_source" and asset_path.is_dir():
            parsed = _parse_traci_method_directory(asset_path, "traci")
            method_frames.append(parsed)
            repo_rows.append(
                {
                    "source_repo": "traci",
                    "asset_path": row.asset_path,
                    "asset_group": row.asset_group,
                    "file_type": row.file_type,
                    "process_rows": 0,
                    "method_rows": len(parsed),
                    "calculable_rows": 0,
                    "notes": row.notes,
                }
            )
        elif row.inferred_role == "flcac_metadata_source":
            repo_rows.append(
                {
                    "source_repo": "federal_lca_commons",
                    "asset_path": row.asset_path,
                    "asset_group": row.asset_group,
                    "file_type": row.file_type,
                    "process_rows": 0,
                    "method_rows": 0,
                    "calculable_rows": 0,
                    "notes": "metadata_or_flow_library_only",
                }
            )

    useeio_assets = audit_frame.loc[audit_frame["inferred_role"] == "useeio_process_source"].copy()
    for row in useeio_assets.itertuples(index=False):
        asset_path = root / str(row.asset_path)
        if asset_path.suffix.lower() == ".xlsx":
            parsed = _parse_useeio_metadata_workbook(asset_path)
            process_frames.append(parsed)
            repo_rows.append(
                {
                    "source_repo": "useeio_repository",
                    "asset_path": row.asset_path,
                    "asset_group": row.asset_group,
                    "file_type": row.file_type,
                    "process_rows": len(parsed),
                    "method_rows": 0,
                    "calculable_rows": int(parsed["calculable_flag"].fillna(False).sum()) if not parsed.empty else 0,
                    "notes": row.notes or "parsed_from_workbook_metadata",
                }
            )

    bridge_assets = audit_frame.loc[audit_frame["inferred_role"] == "bridge_process_source"].copy()
    if bridge_assets.empty:
        repo_rows.append(
            {
                "source_repo": "bridge_processes",
                "asset_path": "data/bridge processes",
                "asset_group": "bridge processes",
                "file_type": "directory",
                "process_rows": 0,
                "method_rows": 0,
                "calculable_rows": 0,
                "notes": "no_bridge_assets_detected",
            }
        )

    standardized_path = (
        root / standardized_process_corpus_path
        if standardized_process_corpus_path is not None and not Path(standardized_process_corpus_path).is_absolute()
        else Path(standardized_process_corpus_path) if standardized_process_corpus_path is not None else root / "data/interim/pv_glass_process_corpus_standardized.csv"
    )
    standardized_frame = _parse_standardized_process_corpus(standardized_path)
    process_frames.append(standardized_frame)
    repo_rows.append(
        {
            "source_repo": "pv_glass_process_corpus_standardized",
            "asset_path": _relative_path(standardized_path, root),
            "asset_group": "interim",
            "file_type": standardized_path.suffix.lower().lstrip("."),
            "process_rows": len(standardized_frame),
            "method_rows": 0,
            "calculable_rows": int(standardized_frame["calculable_flag"].fillna(False).sum()) if not standardized_frame.empty else 0,
            "notes": "existing_standardized_process_corpus",
        }
    )

    process_registry = (
        pd.concat([frame for frame in process_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in process_frames)
        else pd.DataFrame(columns=PROCESS_REGISTRY_COLUMNS)
    )
    if not process_registry.empty:
        process_registry = process_registry.drop_duplicates(
            subset=["source_repo", "process_uuid", "process_name", "asset_path"]
        ).reset_index(drop=True)
    method_registry = (
        pd.concat([frame for frame in method_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in method_frames)
        else pd.DataFrame(columns=METHOD_REGISTRY_COLUMNS)
    )
    if not method_registry.empty:
        method_registry = method_registry.drop_duplicates(subset=["method_uuid", "method_name", "asset_path"]).reset_index(
            drop=True
        )
    repo_registry = pd.DataFrame(repo_rows)
    return repo_registry, process_registry, method_registry


def _first_non_empty(*values: object) -> str:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return ""


def build_pv_glass_reference_registry(repo_root: Path | None = None) -> pd.DataFrame:
    root = _repo_root(repo_root)
    glass_index_path = root / "data/Glass_EPD/index.csv"
    material_index_path = root / "data/Material_EPD/index.csv"
    factor_registry_path = root / "data/interim/glass_factor_registry_standardized.csv"
    case_metadata_path = root / "data/interim/pv_glass_cases_with_metadata.csv"
    baseline_dir = root / "data/glass_baseline"

    glass_index = pd.read_csv(glass_index_path, dtype=str, keep_default_na=False).fillna("") if glass_index_path.exists() else pd.DataFrame()
    material_index = pd.read_csv(material_index_path, dtype=str, keep_default_na=False).fillna("") if material_index_path.exists() else pd.DataFrame()
    factor_registry = (
        pd.read_csv(factor_registry_path, keep_default_na=False).fillna("")
        if factor_registry_path.exists()
        else pd.DataFrame()
    )
    case_metadata = (
        pd.read_csv(case_metadata_path, keep_default_na=False).fillna("")
        if case_metadata_path.exists()
        else pd.DataFrame()
    )

    factor_lookup = {}
    if not factor_registry.empty:
        for row in factor_registry.itertuples(index=False):
            reference_id = _clean_text(getattr(row, "source_id", ""))
            if reference_id:
                factor_lookup[reference_id] = row

    case_lookup = {}
    if not case_metadata.empty and "source_doc_id" in case_metadata.columns:
        case_lookup = (
            case_metadata.sort_values("product_id")
            .groupby("source_doc_id", dropna=False)
            .first()
            .to_dict(orient="index")
        )

    rows: list[dict[str, Any]] = []

    def append_index_rows(frame: pd.DataFrame, source_type_label: str) -> None:
        if frame.empty:
            return
        for row in frame.itertuples(index=False):
            reference_id = _clean_text(getattr(row, "doc_id", "")) or _clean_text(getattr(row, "source_id", ""))
            factor_row = factor_lookup.get(reference_id)
            case_row = case_lookup.get(reference_id, {})
            asset_path = _clean_text(getattr(row, "relative_path", ""))
            factor_value = _clean_optional_float(getattr(factor_row, "factor_value", None) if factor_row else None)
            factor_unit = _clean_text(getattr(factor_row, "factor_unit", "") if factor_row else "")
            thickness_mm = _clean_optional_float(
                _first_non_empty(
                    getattr(factor_row, "thickness_mm", "") if factor_row else "",
                    case_row.get("thickness_mm", ""),
                    extract_thickness_mm(getattr(row, "file_name", "")),
                )
            )
            mass_per_m2 = _clean_optional_float(getattr(factor_row, "mass_per_m2", None) if factor_row else None)
            density_kg_m3 = None
            if thickness_mm and mass_per_m2 and thickness_mm > 0:
                density_kg_m3 = mass_per_m2 / (thickness_mm / 1000.0)
            rows.append(
                {
                    "reference_id": reference_id,
                    "source_type": _first_non_empty(getattr(row, "doc_type", ""), source_type_label),
                    "source_name": _first_non_empty(
                        getattr(factor_row, "process_name", "") if factor_row else "",
                        getattr(row, "manufacturer", ""),
                        clean_filename_title(Path(_clean_text(getattr(row, "file_name", "")) or "unknown.pdf")),
                    ),
                    "product_scope": _first_non_empty(
                        getattr(row, "material_or_product", ""),
                        getattr(factor_row, "stage", "") if factor_row else "",
                    ),
                    "geography": _first_non_empty(
                        getattr(factor_row, "geography", "") if factor_row else "",
                        case_row.get("geography", ""),
                    ),
                    "system_boundary": "",
                    "declared_unit": factor_unit,
                    "factor_value": factor_value,
                    "factor_unit": factor_unit,
                    "thickness_mm": thickness_mm,
                    "density_kg_m3": density_kg_m3,
                    "mass_per_m2": mass_per_m2,
                    "asset_path": asset_path,
                    "note": _first_non_empty(
                        getattr(factor_row, "notes", "") if factor_row else "",
                        getattr(row, "notes", ""),
                        "metadata_only_reference_registry" if factor_value is None else "",
                    ),
                }
            )

    append_index_rows(glass_index, "glass_epd")
    append_index_rows(material_index, "material_epd")

    for pdf_path in sorted(baseline_dir.glob("*.pdf")) if baseline_dir.exists() else []:
        rows.append(
            {
                "reference_id": f"baseline_{pdf_path.stem.lower().replace(' ', '_')}",
                "source_type": "baseline_report",
                "source_name": clean_filename_title(pdf_path),
                "product_scope": "pv_glass_baseline",
                "geography": "",
                "system_boundary": "",
                "declared_unit": "",
                "factor_value": None,
                "factor_unit": "",
                "thickness_mm": _clean_optional_float(extract_thickness_mm(pdf_path.name)),
                "density_kg_m3": None,
                "mass_per_m2": None,
                "asset_path": _relative_path(pdf_path, root),
                "note": "metadata_only_baseline_reference",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=REFERENCE_REGISTRY_COLUMNS)
    frame = frame.drop_duplicates(subset=["reference_id", "asset_path"]).reset_index(drop=True)
    return frame[REFERENCE_REGISTRY_COLUMNS]


def _build_process_query_text(row: pd.Series) -> str:
    return _join_text(
        [
            row.get("title", ""),
            row.get("description", ""),
            row.get("stage_hint", ""),
            row.get("pred_naics_title", ""),
            row.get("pred_naics_code", ""),
            row.get("gold_naics_title", ""),
            row.get("gold_naics_code", ""),
        ]
    )


def _tfidf_rank_records(
    query_frame: pd.DataFrame,
    process_frame: pd.DataFrame,
    top_k: int,
) -> list[dict[str, Any]]:
    vectorizer = TfidfVectorizer(min_df=1)
    corpus_matrix = vectorizer.fit_transform(process_frame["retrieval_text"].astype(str))
    query_matrix = vectorizer.transform(query_frame["query_text"].astype(str))
    score_matrix = linear_kernel(query_matrix, corpus_matrix)
    outputs: list[dict[str, Any]] = []
    for row_index, query_row in enumerate(query_frame.itertuples(index=False)):
        row_scores = score_matrix[row_index]
        top_indices = row_scores.argsort()[::-1][:top_k]
        candidates = []
        for index in top_indices:
            candidate_row = process_frame.iloc[int(index)]
            tfidf_score = float(row_scores[int(index)])
            lexical_score = lexical_overlap_score(query_row.query_text, candidate_row["retrieval_text"])
            candidates.append(
                {
                    "process_uuid": candidate_row.get("process_uuid", ""),
                    "process_name": candidate_row.get("process_name", ""),
                    "retrieval_score": tfidf_score + lexical_score,
                    "rerank_score": tfidf_score + lexical_score,
                    "calculable_flag": _clean_optional_bool(candidate_row.get("calculable_flag", False)),
                    "reference_flow_unit": candidate_row.get("reference_flow_unit", ""),
                    "source_repo": candidate_row.get("source_repo", ""),
                    "asset_path": candidate_row.get("asset_path", ""),
                    "process_category": candidate_row.get("process_category", ""),
                    "location": candidate_row.get("location", ""),
                }
            )
        outputs.append(
            {
                "case_id": query_row.case_id,
                "query_text": query_row.query_text,
                "candidates": candidates,
            }
        )
    return outputs


def retrieve_process_candidates_from_registry(
    cases_frame: pd.DataFrame,
    process_registry: pd.DataFrame,
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    query_frame = cases_frame.copy()
    query_frame["case_id"] = query_frame["product_id"].astype(str)
    query_frame["query_text"] = query_frame.apply(_build_process_query_text, axis=1)

    registry = process_registry.copy()
    registry["retrieval_text"] = registry["retrieval_text"].fillna("")
    if "process_text" not in registry.columns:
        registry["process_text"] = registry["retrieval_text"]
    records = _tfidf_rank_records(query_frame[["case_id", "query_text"]], registry, top_k=top_k)

    top1_rows = []
    for record in records:
        first = record["candidates"][0] if record["candidates"] else {}
        top1_rows.append(
            {
                "case_id": record["case_id"],
                "process_uuid": first.get("process_uuid", ""),
                "process_name": first.get("process_name", ""),
                "retrieval_score": first.get("retrieval_score"),
                "rerank_score": first.get("rerank_score"),
                "calculable_flag": first.get("calculable_flag", False),
                "reference_flow_unit": first.get("reference_flow_unit", ""),
                "source_repo": first.get("source_repo", ""),
            }
        )
    return records, pd.DataFrame(top1_rows)


def _reference_lookup_by_id(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "reference_id" not in frame.columns:
        return {}
    return frame.sort_values("reference_id").groupby("reference_id").first().to_dict(orient="index")


def _reference_lookup_by_asset(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "asset_path" not in frame.columns:
        return {}
    records = (
        frame.loc[frame["asset_path"].astype(str) != ""]
        .sort_values("asset_path")
        .groupby("asset_path")
        .first()
        .to_dict(orient="index")
    )
    return records


def build_calculation_queue_frame(
    retrieval_records: list[dict[str, Any]],
    process_registry: pd.DataFrame,
    case_metadata_frame: pd.DataFrame,
    reference_registry_frame: pd.DataFrame,
    target_unit: str,
) -> pd.DataFrame:
    process_lookup = (
        process_registry.drop_duplicates(subset=["process_uuid", "source_repo", "asset_path"])
        .fillna("")
        .to_dict(orient="records")
    )
    process_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    process_map_no_source: dict[tuple[str, str], dict[str, Any]] = {}
    process_map_by_uuid: dict[str, dict[str, Any]] = {}
    for row in process_lookup:
        process_uuid = _clean_text(row.get("process_uuid", ""))
        process_name = _clean_text(row.get("process_name", ""))
        source_repo = _clean_text(row.get("source_repo", ""))
        key = (process_uuid, process_name, source_repo)
        process_map[key] = row
        process_map_no_source.setdefault((process_uuid, process_name), row)
        if process_uuid:
            process_map_by_uuid.setdefault(process_uuid, row)

    case_lookup = (
        case_metadata_frame.fillna("").sort_values("product_id").groupby("product_id").first().to_dict(orient="index")
        if not case_metadata_frame.empty and "product_id" in case_metadata_frame.columns
        else {}
    )
    reference_by_id = _reference_lookup_by_id(reference_registry_frame)
    reference_by_asset = _reference_lookup_by_asset(reference_registry_frame)

    rows: list[dict[str, Any]] = []
    for record in retrieval_records:
        case_id = _first_non_empty(
            _clean_text(record.get("case_id")),
            _clean_text(record.get("product_id")),
        )
        case_meta = case_lookup.get(case_id, {})
        source_doc_id = _clean_text(case_meta.get("source_doc_id", ""))
        source_file = _clean_text(case_meta.get("source_file", ""))
        reference_meta = reference_by_id.get(source_doc_id) or reference_by_asset.get(source_file) or {}
        for rank, candidate in enumerate(record.get("candidates", []), start=1):
            candidate_process_uuid = _clean_text(candidate.get("process_uuid", ""))
            candidate_process_name = _clean_text(candidate.get("process_name", ""))
            candidate_source_repo = _first_non_empty(
                _clean_text(candidate.get("source_repo", "")),
                _clean_text(candidate.get("source_dataset", "")),
            )
            process_key = (
                candidate_process_uuid,
                candidate_process_name,
                candidate_source_repo,
            )
            process_row = process_map.get(process_key)
            if process_row is None:
                process_row = process_map_no_source.get((candidate_process_uuid, candidate_process_name))
            if process_row is None and candidate_process_uuid:
                process_row = process_map_by_uuid.get(candidate_process_uuid)
            if process_row is None:
                process_row = {}
            reference_flow_unit = _first_non_empty(
                candidate.get("reference_flow_unit", ""),
                process_row.get("reference_flow_unit", ""),
            )
            thickness_mm = _clean_optional_float(
                _first_non_empty(case_meta.get("thickness_mm", ""), reference_meta.get("thickness_mm", ""))
            )
            density_kg_m3 = _clean_optional_float(reference_meta.get("density_kg_m3", None))
            mass_per_m2 = _clean_optional_float(reference_meta.get("mass_per_m2", None))
            if mass_per_m2 is None and thickness_mm and density_kg_m3:
                mass_per_m2 = density_kg_m3 * (thickness_mm / 1000.0)

            blockers = []
            calculable_flag = _clean_optional_bool(
                candidate.get("calculable_flag", process_row.get("calculable_flag", False))
            )
            process_uuid = _first_non_empty(candidate_process_uuid, process_row.get("process_uuid", ""))
            if not calculable_flag:
                blockers.append(_first_non_empty(process_row.get("calc_blocker_reason", ""), "calculable_flag_false"))
            if not process_uuid:
                blockers.append("missing_process_uuid")
            if not reference_flow_unit:
                blockers.append("missing_reference_flow_unit")
            blockers = list(dict.fromkeys(part for part in blockers if part))

            normalization_ready = False
            if _looks_like_m2_unit(reference_flow_unit):
                normalization_ready = True
            elif _looks_like_kg_unit(reference_flow_unit):
                normalization_ready = mass_per_m2 is not None or (thickness_mm is not None and density_kg_m3 is not None)

            calc_status = "queued" if not blockers else "blocked"
            if calc_status == "blocked":
                result_status = "retrieved_only"
            elif normalization_ready:
                result_status = "retrieved_and_calculable"
            else:
                result_status = "retrieved_but_not_normalizable"

            rows.append(
                {
                    "case_id": case_id,
                    "candidate_rank": rank,
                    "process_uuid": process_uuid,
                    "process_name": _first_non_empty(candidate_process_name, process_row.get("process_name", "")),
                    "source_repo": _first_non_empty(candidate_source_repo, process_row.get("source_repo", "")),
                    "asset_path": _first_non_empty(candidate.get("asset_path", ""), process_row.get("asset_path", "")),
                    "retrieval_score": candidate.get("retrieval_score", candidate.get("score")),
                    "rerank_score": candidate.get("rerank_score", candidate.get("score")),
                    "reference_flow_name": _first_non_empty(
                        candidate.get("reference_flow_name", ""),
                        process_row.get("reference_flow_name", ""),
                        candidate_process_name,
                    ),
                    "reference_flow_unit": reference_flow_unit,
                    "target_unit": target_unit,
                    "thickness_mm": thickness_mm,
                    "density_kg_m3": density_kg_m3,
                    "mass_per_m2": mass_per_m2,
                    "geography": _first_non_empty(case_meta.get("geography", ""), reference_meta.get("geography", ""), process_row.get("location", "")),
                    "system_boundary": _first_non_empty(reference_meta.get("system_boundary", ""), case_meta.get("stage_hint", "")),
                    "calc_status": calc_status,
                    "block_reason": ";".join(blockers),
                    "normalization_ready_flag": normalization_ready,
                    "result_status": result_status,
                }
            )
    return pd.DataFrame(rows)


def normalize_climate_result(
    impact_value: float | None,
    impact_unit: str,
    reference_flow_unit: str,
    target_unit: str,
    thickness_mm: float | None = None,
    density_kg_m3: float | None = None,
    mass_per_m2: float | None = None,
) -> NormalizationResult:
    if impact_value is None:
        return NormalizationResult(None, "", "retrieved_only", "missing_impact_value", mass_per_m2)

    normalized_reference = _normalize_unit(reference_flow_unit)
    normalized_target = _normalize_unit(target_unit)
    if normalized_target not in {"kgco2e/m2", "kgco2eq/m2"}:
        return NormalizationResult(None, target_unit, "retrieved_only", "unsupported_target_unit", mass_per_m2)

    computed_mass = mass_per_m2
    if computed_mass is None and thickness_mm is not None and density_kg_m3 is not None:
        computed_mass = density_kg_m3 * (thickness_mm / 1000.0)

    if _looks_like_m2_unit(reference_flow_unit):
        return NormalizationResult(
            float(impact_value),
            "kgCO2e/m2",
            "retrieved_and_calculable",
            "already_per_m2",
            computed_mass,
        )

    if _looks_like_kg_unit(reference_flow_unit):
        if computed_mass is None:
            return NormalizationResult(
                None,
                "kgCO2e/m2",
                "retrieved_but_not_normalizable",
                "missing_mass_per_m2",
                None,
            )
        return NormalizationResult(
            float(impact_value) * float(computed_mass),
            "kgCO2e/m2",
            "retrieved_and_calculable",
            "converted_from_kg_basis",
            computed_mass,
        )

    return NormalizationResult(
        None,
        "kgCO2e/m2",
        "retrieved_but_not_normalizable",
        f"unsupported_reference_flow_unit:{reference_flow_unit}",
        computed_mass,
    )


def _impact_is_climate_indicator(category_name: str, impact_unit: str) -> bool:
    category = _clean_text(category_name).lower()
    unit = _normalize_unit(impact_unit)
    return "global warming" in category or "climate change" in category or unit.startswith("kgco2e")


def run_openlca_calculations(
    queue_frame: pd.DataFrame,
    method_name: str,
    output_dir: Path,
    port: int = 8080,
    timeout_sec: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_directory(output_dir)
    raw_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    raw_columns = [
        "case_id",
        "process_uuid",
        "process_name",
        "source_repo",
        "reference_flow_unit",
        "impact_method_name",
        "impact_category_name",
        "impact_unit",
        "impact_value",
        "target_unit",
        "thickness_mm",
        "density_kg_m3",
        "mass_per_m2",
        "calc_status",
    ]
    normalized_columns = [
        "case_id",
        "process_uuid",
        "process_name",
        "source_repo",
        "impact_method_name",
        "impact_category_name",
        "impact_unit",
        "impact_value",
        "reference_flow_unit",
        "target_unit",
        "normalized_value",
        "normalized_unit",
        "normalization_note",
        "mass_per_m2",
        "result_status",
    ]
    failure_columns = ["case_id", "process_uuid", "process_name", "failure_reason", "failure_trace"]

    def finalize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raw_frame = pd.DataFrame(raw_rows, columns=raw_columns)
        normalized_frame = pd.DataFrame(normalized_rows, columns=normalized_columns)
        failure_frame = pd.DataFrame(failure_rows, columns=failure_columns)
        raw_frame.to_csv(output_dir / "openlca_impacts_raw.csv", index=False)
        normalized_frame.to_csv(output_dir / "openlca_impacts_normalized.csv", index=False)
        failure_frame.to_csv(output_dir / "openlca_calc_failures.csv", index=False)
        return raw_frame, normalized_frame, failure_frame

    queued = queue_frame.loc[queue_frame["calc_status"].astype(str) == "queued"].copy() if not queue_frame.empty else pd.DataFrame()
    if queued.empty:
        return finalize()

    try:
        import olca
    except Exception as exc:
        for row in queued.itertuples(index=False):
            failure_rows.append(
                {
                    "case_id": row.case_id,
                    "process_uuid": row.process_uuid,
                    "process_name": row.process_name,
                    "failure_reason": f"olca_import_error: {exc}",
                }
            )
        return finalize()

    ipc_url = f"http://127.0.0.1:{port}"
    descriptor_cache: dict[str, list[dict[str, Any]]] = {}
    product_system_cache: dict[str, dict[str, Any]] = {}

    def ipc_post(method: str, params: dict[str, Any]) -> Any:
        response = requests.post(
            ipc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        error = payload.get("error")
        if error:
            raise RuntimeError(str(error.get("message", error)))
        return payload.get("result")

    def model_type_name(model_type: Any) -> str:
        if isinstance(model_type, str):
            return model_type
        return str(getattr(model_type, "__name__", model_type))

    def get_descriptors(model_type: Any) -> list[dict[str, Any]]:
        type_name = model_type_name(model_type)
        cached = descriptor_cache.get(type_name)
        if cached is None:
            result = ipc_post("data/get/descriptors", {"@type": type_name})
            cached = result if isinstance(result, list) else []
            descriptor_cache[type_name] = cached
        return cached

    def resolve_ref(model_type: Any, uid: str = "", name: str = "") -> dict[str, Any] | None:
        type_name = model_type_name(model_type)
        if uid:
            try:
                entity = ipc_post("data/get", {"@type": type_name, "@id": uid})
            except Exception:
                entity = None
            if isinstance(entity, dict):
                entity_id = _clean_text(entity.get("@id", ""))
                entity_name = _clean_text(entity.get("name", "")) or name
                if entity_id:
                    ref = {"@type": type_name, "@id": entity_id}
                    if entity_name:
                        ref["name"] = entity_name
                    return ref

        try:
            descriptors = get_descriptors(model_type)
        except Exception:
            descriptors = []

        if uid:
            for descriptor in descriptors:
                descriptor_id = _clean_text(descriptor.get("@id", ""))
                descriptor_name = _clean_text(descriptor.get("name", ""))
                if descriptor_id == uid:
                    ref = {"@type": type_name, "@id": descriptor_id}
                    if descriptor_name or name:
                        ref["name"] = descriptor_name or name
                    return ref

        if name:
            normalized_name = _clean_text(name)
            for descriptor in descriptors:
                descriptor_id = _clean_text(descriptor.get("@id", ""))
                descriptor_name = _clean_text(descriptor.get("name", ""))
                if descriptor_name == normalized_name and descriptor_id:
                    ref = {"@type": type_name, "@id": descriptor_id}
                    if descriptor_name:
                        ref["name"] = descriptor_name
                    return ref

        return None

    try:
        method_ref = resolve_ref(olca.ImpactMethod, name=method_name)
        if method_ref is None:
            raise RuntimeError(f"impact method not found in openLCA IPC database: {method_name}")
    except Exception as exc:
        for row in queued.itertuples(index=False):
            failure_rows.append(
                {
                    "case_id": row.case_id,
                    "process_uuid": row.process_uuid,
                    "process_name": row.process_name,
                    "failure_reason": f"openlca_connection_or_method_error: {exc}",
                }
            )
        return finalize()

    for row in queued.itertuples(index=False):
        result_id = ""
        try:
            process_ref = resolve_ref(olca.Process, uid=str(row.process_uuid))
            if process_ref is None:
                raise RuntimeError("process_uuid_not_found_in_openlca_db")

            product_system_ref = product_system_cache.get(str(row.process_uuid))
            if product_system_ref is None:
                product_system_ref = ipc_post(
                    "data/create/system",
                    {
                        "process": process_ref,
                        "config": {
                            "providerLinking": "prefer",
                            "preferUnitProcesses": False,
                        },
                    },
                )
                if not isinstance(product_system_ref, dict) or not _clean_text(product_system_ref.get("@id", "")):
                    raise RuntimeError("product_system_creation_failed")
                product_system_cache[str(row.process_uuid)] = product_system_ref
            if product_system_ref is None:
                raise RuntimeError("product_system_creation_failed")

            calculation_state = ipc_post(
                "result/calculate",
                {
                    "@type": "CalculationSetup",
                    "calculationType": "SIMPLE_CALCULATION",
                    "productSystem": product_system_ref,
                    "impactMethod": method_ref,
                    "amount": 1.0,
                },
            )
            if not isinstance(calculation_state, dict):
                raise RuntimeError("calculation_state_missing")
            result_id = _clean_text(calculation_state.get("@id", "") or calculation_state.get("id", ""))
            if not result_id:
                raise RuntimeError("calculation_result_id_missing")

            deadline = time.time() + timeout_sec
            result_state = calculation_state
            while not bool(result_state.get("isReady")):
                if time.time() >= deadline:
                    raise RuntimeError("calculation_timeout")
                time.sleep(1.0)
                result_state = ipc_post("result/state", {"@id": result_id})
                if not isinstance(result_state, dict):
                    raise RuntimeError("calculation_state_poll_failed")

            impact_results = ipc_post("result/total-impacts", {"@id": result_id})
            if not isinstance(impact_results, list):
                raise RuntimeError("impact_results_missing")
            for impact in impact_results:
                category_ref = impact.get("impactCategory", {}) if isinstance(impact, dict) else {}
                category_name = _clean_text(category_ref.get("name", ""))
                impact_unit = _clean_text(category_ref.get("refUnit", ""))
                impact_value = float(impact.get("amount", 0.0)) if isinstance(impact, dict) else 0.0
                raw_rows.append(
                    {
                        "case_id": row.case_id,
                        "process_uuid": row.process_uuid,
                        "process_name": row.process_name,
                        "source_repo": row.source_repo,
                        "reference_flow_unit": row.reference_flow_unit,
                        "impact_method_name": method_name,
                        "impact_category_name": category_name,
                        "impact_unit": impact_unit,
                        "impact_value": impact_value,
                        "target_unit": row.target_unit,
                        "thickness_mm": row.thickness_mm,
                        "density_kg_m3": row.density_kg_m3,
                        "mass_per_m2": row.mass_per_m2,
                        "calc_status": "success",
                    }
                )
                if _impact_is_climate_indicator(category_name, impact_unit):
                    normalization = normalize_climate_result(
                        impact_value=impact_value,
                        impact_unit=impact_unit,
                        reference_flow_unit=str(row.reference_flow_unit),
                        target_unit=str(row.target_unit),
                        thickness_mm=_clean_optional_float(row.thickness_mm),
                        density_kg_m3=_clean_optional_float(row.density_kg_m3),
                        mass_per_m2=_clean_optional_float(row.mass_per_m2),
                    )
                    normalized_rows.append(
                        {
                            "case_id": row.case_id,
                            "process_uuid": row.process_uuid,
                            "process_name": row.process_name,
                            "source_repo": row.source_repo,
                            "impact_method_name": method_name,
                            "impact_category_name": category_name,
                            "impact_unit": impact_unit,
                            "impact_value": impact_value,
                            "reference_flow_unit": row.reference_flow_unit,
                            "target_unit": row.target_unit,
                            "normalized_value": normalization.normalized_value,
                            "normalized_unit": normalization.normalized_unit,
                            "normalization_note": normalization.note,
                            "mass_per_m2": normalization.mass_per_m2,
                            "result_status": normalization.result_status,
                        }
                    )
        except Exception as exc:
            failure_rows.append(
                {
                    "case_id": row.case_id,
                    "process_uuid": row.process_uuid,
                    "process_name": row.process_name,
                    "failure_reason": str(exc),
                    "failure_trace": traceback.format_exc(limit=5),
                }
            )
        finally:
            try:
                if result_id:
                    ipc_post("result/dispose", {"@id": result_id})
            except Exception:
                pass

    return finalize()


def write_registry_outputs(
    repo_registry: pd.DataFrame,
    process_registry: pd.DataFrame,
    method_registry: pd.DataFrame,
    repo_registry_csv: str | Path,
    process_registry_parquet: str | Path,
    method_registry_parquet: str | Path,
) -> None:
    repo_path = Path(repo_registry_csv)
    process_path = Path(process_registry_parquet)
    method_path = Path(method_registry_parquet)
    ensure_directory(repo_path.parent)
    ensure_directory(process_path.parent)
    ensure_directory(method_path.parent)
    repo_registry.to_csv(repo_path, index=False)
    write_parquet(process_registry, process_path)
    write_parquet(method_registry, method_path)
