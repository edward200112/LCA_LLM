from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from open_match_lca.constants import PROJECT_ROOT
from open_match_lca.data.external_assets import EXTERNAL_ASSETS, fetch_external_assets
from open_match_lca.data.pv_glass_extension import (
    FACTOR_REGISTRY_COLUMNS,
    PROCESS_CORPUS_COLUMNS,
    build_pv_glass_cases,
    build_standardized_epa_factors,
    build_enriched_naics,
    write_pv_glass_config,
)
from open_match_lca.io_utils import ensure_directory, load_yaml
from open_match_lca.schemas import normalize_naics_code


RAW_EPA_STANDARD_COLUMNS = [
    "naics_code",
    "factor_value",
    "factor_unit",
    "with_margins",
    "without_margins",
    "source_year",
    "useeio_code",
]
RAW_NAICS_STANDARD_COLUMNS = [
    "naics_code",
    "naics_title",
    "naics_description",
]
GLASS_REGISTRY_REQUIRED_COLUMNS = FACTOR_REGISTRY_COLUMNS
PROCESS_CORPUS_REQUIRED_COLUMNS = PROCESS_CORPUS_COLUMNS


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or PROJECT_ROOT).resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def _missing_rates(frame: pd.DataFrame) -> dict[str, float]:
    rates: dict[str, float] = {}
    if frame.empty:
        return {column: 1.0 for column in frame.columns}
    total = len(frame)
    for column in frame.columns:
        missing = frame[column].astype(str).str.strip().eq("").sum()
        rates[column] = round(float(missing / total), 4)
    return rates


def _preview(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "readable": False, "columns": [], "row_count": 0}
    if path.suffix.lower() == ".csv":
        frame = _read_csv(path)
        return {
            "exists": True,
            "size_bytes": path.stat().st_size,
            "readable": True,
            "columns": list(frame.columns),
            "row_count": int(len(frame)),
        }
    if path.suffix.lower() == ".xlsx":
        try:
            frame = pd.read_excel(path, nrows=5)
            return {
                "exists": True,
                "size_bytes": path.stat().st_size,
                "readable": True,
                "columns": list(frame.columns),
                "row_count": -1,
            }
        except Exception:
            return {"exists": True, "size_bytes": path.stat().st_size, "readable": False, "columns": [], "row_count": -1}
    return {"exists": True, "size_bytes": path.stat().st_size, "readable": True, "columns": [], "row_count": -1}


def inspect_raw_sources(repo_root: Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo_root)
    epa_dir = root / "data" / "raw" / "epa_factors"
    naics_dir = root / "data" / "raw" / "naics"
    external_dir = root / "data" / "external_downloads"

    epa_official_candidates = [
        external_dir / "epa" / "SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv",
        epa_dir / "SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv",
    ]
    naics_2017_candidates = [
        external_dir / "naics" / "2017_NAICS_Structure.xlsx",
        naics_dir / "2017_NAICS_Structure.xlsx",
    ]
    naics_2022_candidates = [
        external_dir / "naics" / "2022_NAICS_Structure.xlsx",
        naics_dir / "2022_NAICS_Structure.xlsx",
    ]
    concordance_candidates = [
        external_dir / "naics" / "2022_to_2017_NAICS.xlsx",
        naics_dir / "2022_to_2017_NAICS.xlsx",
    ]

    epa_standard_path = epa_dir / "epa_naics_v13.csv"
    epa_cleaned_path = epa_dir / "epa_factors_from_caml.csv"
    naics_cleaned_path = naics_dir / "naics_from_caml.csv"
    naics_enriched_path = naics_dir / "naics_2017_enriched.csv"

    epa_cleaned = _read_csv(epa_cleaned_path) if epa_cleaned_path.exists() else pd.DataFrame()
    epa_standard = _read_csv(epa_standard_path) if epa_standard_path.exists() else pd.DataFrame()
    naics_cleaned = _read_csv(naics_cleaned_path) if naics_cleaned_path.exists() else pd.DataFrame()
    naics_enriched = _read_csv(naics_enriched_path) if naics_enriched_path.exists() else pd.DataFrame()

    return {
        "epa_dir_files": sorted(path.name for path in epa_dir.glob("*") if path.is_file()),
        "naics_dir_files": sorted(path.name for path in naics_dir.glob("*") if path.is_file()),
        "official_epa_raw": next((path for path in epa_official_candidates if path.exists() and path.stat().st_size > 0), None),
        "official_naics_2017": next((path for path in naics_2017_candidates if path.exists() and path.stat().st_size > 0), None),
        "official_naics_2022": next((path for path in naics_2022_candidates if path.exists() and path.stat().st_size > 0), None),
        "official_naics_concordance": next((path for path in concordance_candidates if path.exists() and path.stat().st_size > 0), None),
        "epa_standard_path": epa_standard_path,
        "naics_enriched_path": naics_enriched_path,
        "epa_cleaned_path": epa_cleaned_path,
        "naics_cleaned_path": naics_cleaned_path,
        "epa_standard_valid": list(epa_standard.columns) == RAW_EPA_STANDARD_COLUMNS and not epa_standard.empty,
        "naics_enriched_valid": all(column in naics_enriched.columns for column in RAW_NAICS_STANDARD_COLUMNS[:2]) and not naics_enriched.empty,
        "epa_cleaned_valid": list(epa_cleaned.columns) == RAW_EPA_STANDARD_COLUMNS and not epa_cleaned.empty,
        "naics_cleaned_valid": all(column in naics_cleaned.columns for column in RAW_NAICS_STANDARD_COLUMNS[:2]) and not naics_cleaned.empty,
        "epa_standard_preview": _preview(epa_standard_path),
        "naics_enriched_preview": _preview(naics_enriched_path),
        "epa_cleaned_preview": _preview(epa_cleaned_path),
        "naics_cleaned_preview": _preview(naics_cleaned_path),
        "official_epa_preview": _preview(next((path for path in epa_official_candidates if path.exists()), epa_official_candidates[0])),
        "official_naics_2017_preview": _preview(next((path for path in naics_2017_candidates if path.exists()), naics_2017_candidates[0])),
        "official_naics_2022_preview": _preview(next((path for path in naics_2022_candidates if path.exists()), naics_2022_candidates[0])),
        "official_concordance_preview": _preview(next((path for path in concordance_candidates if path.exists()), concordance_candidates[0])),
        "epa_duplicate_naics": int(epa_standard["naics_code"].duplicated().sum()) if not epa_standard.empty else 0,
        "naics_duplicate_codes": int(naics_enriched["naics_code"].duplicated().sum()) if not naics_enriched.empty else 0,
    }


def ensure_stage_a_assets(
    repo_root: Path | None = None,
    *,
    force: bool = False,
    allow_download: bool = True,
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    inspection_before = inspect_raw_sources(root)
    downloads_frame = pd.DataFrame()
    missing_official = inspection_before["official_epa_raw"] is None or inspection_before["official_naics_2017"] is None
    download_log_path = root / "reports" / "audit" / "download_log.csv"
    prior_download_log = _read_csv(download_log_path) if download_log_path.exists() else pd.DataFrame()
    prior_attempted_asset_ids = set(prior_download_log.get("asset_id", pd.Series(dtype=str)).astype(str).tolist())
    required_asset_ids = {
        "epa_supply_chain_v13",
        "naics_2017_structure",
        "naics_2022_structure",
        "naics_2022_to_2017_concordance",
    }
    should_retry_downloads = force or not required_asset_ids.issubset(prior_attempted_asset_ids)
    if allow_download and missing_official and should_retry_downloads:
        downloads_frame = fetch_external_assets(root, force=force, include_optional=True)
    epa_path = build_standardized_epa_factors(root, force=force) if force or not inspection_before["epa_standard_valid"] else inspection_before["epa_standard_path"]
    naics_path = build_enriched_naics(root, force=force) if force or not inspection_before["naics_enriched_valid"] else inspection_before["naics_enriched_path"]
    inspection_after = inspect_raw_sources(root)
    return {
        "inspection_before": inspection_before,
        "inspection_after": inspection_after,
        "downloads": downloads_frame,
        "epa_standard_path": Path(epa_path),
        "naics_enriched_path": Path(naics_path),
    }


def write_stage_a_report(repo_root: Path | None = None, *, force: bool = False, allow_download: bool = True) -> Path:
    root = _repo_root(repo_root)
    results = ensure_stage_a_assets(root, force=force, allow_download=allow_download)
    inspection = results["inspection_after"]
    download_log_path = root / "reports" / "audit" / "download_log.csv"
    download_log = _read_csv(download_log_path) if download_log_path.exists() else pd.DataFrame(columns=["asset_id", "status", "error"])

    reused = []
    generated = []
    missing = []
    downloaded = []

    if inspection["epa_cleaned_valid"]:
        reused.append("`data/raw/epa_factors/epa_factors_from_caml.csv` as an existing parser-ready EPA table")
    if inspection["naics_cleaned_valid"]:
        reused.append("`data/raw/naics/naics_from_caml.csv` as an existing cleaned NAICS table")
    if inspection["epa_standard_valid"] and inspection["epa_standard_path"].exists():
        generated.append("`data/raw/epa_factors/epa_naics_v13.csv`")
    if inspection["naics_enriched_valid"] and inspection["naics_enriched_path"].exists():
        generated.append("`data/raw/naics/naics_2017_enriched.csv`")

    if inspection["official_epa_raw"] is None:
        missing.append("official EPA v1.3 raw CSV")
    if inspection["official_naics_2017"] is None:
        missing.append("official 2017 NAICS structure workbook")
    if inspection["official_naics_2022"] is None:
        missing.append("official 2022 NAICS structure workbook")
    if inspection["official_naics_concordance"] is None:
        missing.append("2022_to_2017 concordance workbook")

    if not download_log.empty:
        for row in download_log.to_dict(orient="records"):
            if row.get("status") == "downloaded":
                downloaded.append(f"`{row['asset_id']}`")

    lines = [
        "# Stage A Source Standardization",
        "",
        "## Raw Source Inspection",
        "",
        f"- `data/raw/epa_factors` files: {', '.join(inspection['epa_dir_files']) or '(none)'}.",
        f"- `data/raw/naics` files: {', '.join(inspection['naics_dir_files']) or '(none)'}.",
        f"- Official EPA raw CSV present: `{bool(inspection['official_epa_raw'])}`.",
        f"- Official 2017 NAICS workbook present: `{bool(inspection['official_naics_2017'])}`.",
        f"- Official 2022 NAICS workbook present: `{bool(inspection['official_naics_2022'])}`.",
        f"- Official 2022->2017 concordance present: `{bool(inspection['official_naics_concordance'])}`.",
        "",
        "## Reused Raw Files",
        "",
    ]
    lines.extend(f"- {item}" for item in (reused or ["No reusable raw files were detected."]))
    lines.extend(
        [
            "",
            "## Download Decisions",
            "",
            f"- Download log path: `{download_log_path.relative_to(root)}`.",
        ]
    )
    if downloaded:
        lines.extend(f"- Downloaded {item}." for item in downloaded)
    else:
        lines.append("- No new official source downloads were completed in this stage.")
    if not download_log.empty:
        failures = download_log.loc[download_log["status"].astype(str).str.contains("failed", na=False)]
        for row in failures.to_dict(orient="records"):
            lines.append(f"- Download failure: `{row['asset_id']}` -> `{row['error']}`")

    lines.extend(
        [
            "",
            "## Standardized Copies",
            "",
            f"- EPA standardized schema valid: `{inspection['epa_standard_valid']}` with columns {inspection['epa_standard_preview']['columns']}.",
            f"- NAICS enriched schema valid: `{inspection['naics_enriched_valid']}` with columns {inspection['naics_enriched_preview']['columns']}.",
            "- EPA duplicate handling rule: sort by `naics_code`, `source_year` descending, then `factor_value` descending, and keep the first row per `naics_code` when rebuilding.",
        ]
    )
    lines.extend(f"- Generated or reused {item}." for item in generated)
    lines.extend(
        [
            "",
            "## Still Missing",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in (missing or ["No missing Stage A source artifacts remain."]))

    out_path = root / "reports" / "audit" / "stage_a_source_standardization.md"
    ensure_directory(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _canonical_source_type(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"epd", "other", "standard", "report", "spec", "tds"}:
        return text
    return text or "other"


def _canonical_stage(value: str) -> str:
    text = str(value).strip().lower()
    mapping = {
        "flat_glass": "flat_glass",
        "tempering_or_finishing": "tempering_or_finishing",
        "raw_material": "raw_material",
        "unknown": "unknown",
    }
    return mapping.get(text, text or "unknown")


def _normalize_float_text(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return ""
    if number.is_integer():
        return str(int(number))
    return str(number)


def audit_glass_registry(repo_root: Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo_root)
    path = root / "data" / "interim" / "glass_factor_registry.csv"
    frame = _read_csv(path)
    missing_columns = [column for column in GLASS_REGISTRY_REQUIRED_COLUMNS if column not in frame.columns]
    summary = {
        "path": path,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "missing_columns": missing_columns,
        "missing_rates": _missing_rates(frame),
        "duplicate_source_id": int(frame["source_id"].duplicated().sum()) if "source_id" in frame.columns else -1,
        "factor_unit_distribution": frame.get("factor_unit", pd.Series(dtype=str)).replace("", "<empty>").value_counts().to_dict(),
        "stage_distribution": frame.get("stage", pd.Series(dtype=str)).replace("", "<empty>").value_counts().to_dict(),
        "source_type_distribution": frame.get("source_type", pd.Series(dtype=str)).replace("", "<empty>").value_counts().to_dict(),
        "geography_distribution": frame.get("geography", pd.Series(dtype=str)).replace("", "<empty>").value_counts().to_dict(),
        "numeric_factor_values": int(pd.to_numeric(frame.get("factor_value", pd.Series(dtype=str)), errors="coerce").notna().sum()) if "factor_value" in frame.columns else 0,
    }
    return summary


def standardize_glass_registry(repo_root: Path | None = None, *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    root = _repo_root(repo_root)
    input_path = root / "data" / "interim" / "glass_factor_registry.csv"
    output_path = root / "data" / "interim" / "glass_factor_registry_standardized.csv"
    frame = _read_csv(input_path)
    for column in GLASS_REGISTRY_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    standardized = frame[GLASS_REGISTRY_REQUIRED_COLUMNS].copy()
    for column in standardized.columns:
        standardized[column] = standardized[column].astype(str).str.strip()
    standardized["source_type"] = standardized["source_type"].map(_canonical_source_type)
    standardized["stage"] = standardized["stage"].map(_canonical_stage)
    standardized["naics_code"] = standardized["naics_code"].map(normalize_naics_code)
    standardized["naics_code"] = standardized["naics_code"].mask(standardized["naics_code"].eq("000000"), "")
    for column in ["factor_value", "thickness_mm", "mass_per_m2"]:
        standardized[column] = standardized[column].map(_normalize_float_text)
    standardized["year"] = standardized["year"].map(lambda value: str(int(float(value))) if str(value).strip().replace(".", "", 1).isdigit() else "")
    standardized["quality_tier"] = standardized["quality_tier"].replace("", "metadata_only")
    standardized = standardized.drop_duplicates(subset=["source_id"], keep="first").sort_values("source_id").reset_index(drop=True)
    if force or not output_path.exists():
        standardized.to_csv(output_path, index=False, encoding="utf-8")
    audit = audit_glass_registry(root)
    audit["standardized_output"] = output_path
    audit["standardized_row_count"] = int(len(standardized))
    return output_path, audit


def _augment_process_text(row: pd.Series) -> str:
    text = str(row.get("process_text", "")).strip()
    token_count = len(text.split())
    extras = []
    if row.get("process_name"):
        extras.append(f"Process: {row['process_name']}.")
    if row.get("category_path"):
        extras.append(f"Category: {row['category_path']}.")
    if row.get("geography"):
        extras.append(f"Geography: {row['geography']}.")
    if row.get("reference_flow_name"):
        unit = f" ({row['reference_flow_unit']})" if row.get("reference_flow_unit") else ""
        extras.append(f"Reference flow: {row['reference_flow_name']}{unit}.")
    if row.get("source_type"):
        extras.append(f"Source type: {row['source_type']}.")
    if token_count < 20:
        text = " ".join(part for part in [text, *extras] if part).strip()
    return " ".join(text.split())


def audit_process_corpus(repo_root: Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo_root)
    path = root / "data" / "interim" / "pv_glass_process_corpus.csv"
    frame = _read_csv(path)
    missing_columns = [column for column in PROCESS_CORPUS_REQUIRED_COLUMNS if column not in frame.columns]
    process_text_lengths = frame.get("process_text", pd.Series(dtype=str)).map(lambda value: len(str(value).split()))
    summary = {
        "path": path,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "missing_columns": missing_columns,
        "missing_rates": _missing_rates(frame),
        "duplicate_process_uuid": int(frame["process_uuid"].duplicated().sum()) if "process_uuid" in frame.columns else -1,
        "avg_process_text_len": round(float(process_text_lengths.mean()), 2) if not process_text_lengths.empty else 0.0,
        "source_type_distribution": frame.get("source_type", pd.Series(dtype=str)).replace("", "<empty>").value_counts().to_dict(),
        "geography_distribution": frame.get("geography", pd.Series(dtype=str)).replace("", "<empty>").value_counts().to_dict(),
        "reference_flow_unit_distribution": frame.get("reference_flow_unit", pd.Series(dtype=str)).replace("", "<empty>").value_counts().to_dict(),
    }
    return summary


def standardize_process_corpus(repo_root: Path | None = None, *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    root = _repo_root(repo_root)
    input_path = root / "data" / "interim" / "pv_glass_process_corpus.csv"
    output_path = root / "data" / "interim" / "pv_glass_process_corpus_standardized.csv"
    frame = _read_csv(input_path)
    for column in PROCESS_CORPUS_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    standardized = frame[PROCESS_CORPUS_REQUIRED_COLUMNS].copy()
    for column in standardized.columns:
        standardized[column] = standardized[column].astype(str).str.strip()
    standardized["source_type"] = standardized["source_type"].map(_canonical_source_type)
    standardized["process_text"] = standardized.apply(_augment_process_text, axis=1)
    standardized = standardized.drop_duplicates(subset=["process_uuid"], keep="first").sort_values("process_uuid").reset_index(drop=True)
    if force or not output_path.exists():
        standardized.to_csv(output_path, index=False, encoding="utf-8")
    audit = audit_process_corpus(root)
    audit["standardized_output"] = output_path
    audit["standardized_row_count"] = int(len(standardized))
    audit["standardized_avg_process_text_len"] = round(float(standardized["process_text"].map(lambda value: len(str(value).split())).mean()), 2)
    return output_path, audit


def audit_pv_glass_cases(repo_root: Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo_root)
    raw_path = root / "data" / "raw" / "amazon_caml" / "pv_glass_cases.csv"
    metadata_path = root / "data" / "interim" / "pv_glass_cases_with_metadata.csv"
    raw_frame = _read_csv(raw_path)
    metadata_frame = _read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    class_distribution = raw_frame.get("gold_naics_code", pd.Series(dtype=str)).value_counts().to_dict()
    stage_distribution = metadata_frame.get("stage_hint", pd.Series(dtype=str)).value_counts().to_dict() if not metadata_frame.empty else {}
    return {
        "raw_path": raw_path,
        "metadata_path": metadata_path,
        "row_count": int(len(raw_frame)),
        "columns": list(raw_frame.columns),
        "class_distribution": class_distribution,
        "stage_distribution": stage_distribution,
        "metadata_columns": list(metadata_frame.columns) if not metadata_frame.empty else [],
    }


def ensure_stage_b_assets(repo_root: Path | None = None, *, force: bool = False) -> dict[str, Any]:
    root = _repo_root(repo_root)
    registry_std_path, registry_audit = standardize_glass_registry(root, force=force)
    process_std_path, process_audit = standardize_process_corpus(root, force=force)
    pv_cases_raw_path, pv_cases_meta_path = build_pv_glass_cases(root, force=force)
    cases_audit = audit_pv_glass_cases(root)
    return {
        "registry_standardized_path": registry_std_path,
        "process_standardized_path": process_std_path,
        "registry_audit": registry_audit,
        "process_audit": process_audit,
        "cases_audit": cases_audit,
        "pv_cases_raw_path": pv_cases_raw_path,
        "pv_cases_meta_path": pv_cases_meta_path,
    }


def write_stage_b_report(repo_root: Path | None = None, *, force: bool = False) -> Path:
    root = _repo_root(repo_root)
    results = ensure_stage_b_assets(root, force=force)
    registry_audit = results["registry_audit"]
    process_audit = results["process_audit"]
    cases_audit = results["cases_audit"]
    registry_missing = registry_audit["missing_columns"]
    process_missing = process_audit["missing_columns"]
    lines = [
        "# Stage B Glass Extension Audit",
        "",
        "## Registry Audit",
        "",
        f"- Original file: `{Path(registry_audit['path']).relative_to(root)}`.",
        f"- Row count: `{registry_audit['row_count']}`.",
        f"- Columns: {', '.join(f'`{column}`' for column in registry_audit['columns'])}.",
        f"- Missing required columns: {', '.join(registry_missing) if registry_missing else '(none)'}.",
        f"- Duplicate `source_id` rows: `{registry_audit['duplicate_source_id']}`.",
        f"- `factor_unit` distribution: `{registry_audit['factor_unit_distribution']}`.",
        f"- `stage` distribution: `{registry_audit['stage_distribution']}`.",
        f"- `source_type` distribution: `{registry_audit['source_type_distribution']}`.",
        f"- `geography` distribution: `{registry_audit['geography_distribution']}`.",
        f"- Numeric `factor_value` rows: `{registry_audit['numeric_factor_values']}`.",
        f"- Standardized copy: `{Path(results['registry_standardized_path']).relative_to(root)}`.",
        "",
        "## Process Corpus Audit",
        "",
        f"- Original file: `{Path(process_audit['path']).relative_to(root)}`.",
        f"- Row count: `{process_audit['row_count']}`.",
        f"- Columns: {', '.join(f'`{column}`' for column in process_audit['columns'])}.",
        f"- Missing required columns: {', '.join(process_missing) if process_missing else '(none)'}.",
        f"- Duplicate `process_uuid` rows: `{process_audit['duplicate_process_uuid']}`.",
        f"- Average `process_text` length: `{process_audit['avg_process_text_len']}` tokens before standardization, `{process_audit['standardized_avg_process_text_len']}` after standardization.",
        f"- `source_type` distribution: `{process_audit['source_type_distribution']}`.",
        f"- `geography` distribution: `{process_audit['geography_distribution']}`.",
        f"- `reference_flow_unit` distribution: `{process_audit['reference_flow_unit_distribution']}`.",
        f"- Standardized copy: `{Path(results['process_standardized_path']).relative_to(root)}`.",
        "",
        "## PV Glass Cases",
        "",
        f"- Cases file: `{Path(cases_audit['raw_path']).relative_to(root)}`.",
        f"- Metadata file: `{Path(cases_audit['metadata_path']).relative_to(root)}`.",
        f"- Total cases: `{cases_audit['row_count']}`.",
        f"- NAICS class distribution: `{cases_audit['class_distribution']}`.",
        f"- Stage-hint distribution: `{cases_audit['stage_distribution']}`.",
        "",
        "## Remaining Manual Gaps",
        "",
        "- `glass_factor_registry.csv` still has sparse `factor_value`, `factor_unit`, `geography`, and `year` coverage and should be manually enriched from source tables.",
        "- `pv_glass_process_corpus.csv` now has a standardized retrieval-text copy, but the auxiliary PDF-derived rows still rely on lightweight metadata rather than rich born-digital table extraction.",
        "- `pv_glass_cases_with_metadata.csv` still has mostly empty `geography`; this should be backfilled only where sources state it clearly.",
    ]
    out_path = root / "reports" / "audit" / "stage_b_glass_extension_audit.md"
    ensure_directory(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def ensure_stage_c_assets(repo_root: Path | None = None, *, force: bool = False) -> dict[str, Path]:
    root = _repo_root(repo_root)
    config_path = write_pv_glass_config(root, force=force)
    config = load_yaml(config_path)
    updated = False
    for key, value in {
        "pv_cases_path": "data/processed/pv_glass_cases.parquet",
        "glass_registry_path": "data/interim/glass_factor_registry_standardized.csv",
        "glass_process_corpus_path": "data/interim/pv_glass_process_corpus_standardized.csv",
    }.items():
        if config.get(key) != value:
            config[key] = value
            updated = True
    if updated or force:
        Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {"config_path": Path(config_path)}


def write_stage_c_run_plan(repo_root: Path | None = None, *, force: bool = False) -> Path:
    root = _repo_root(repo_root)
    assets = ensure_stage_c_assets(root, force=force)
    config_path = assets["config_path"]
    config = load_yaml(config_path)
    processed_present = (root / "data" / "processed" / "products.parquet").exists()
    splits_present = (root / "data" / "splits" / "random_stratified_train.parquet").exists()
    uslci_present = (root / "data" / "processed" / "uslci_processes.parquet").exists()

    lines = [
        "# Stage C Run Plan",
        "",
        "## Config Decision",
        "",
        f"- Case-study config: `{config_path.relative_to(root)}`.",
        f"- `whether_process_extension`: `{config.get('whether_process_extension')}`.",
        "- `configs/exp/full.yaml` was left untouched because the current orchestration eagerly resolves `uslci_path` and `data/processed/uslci_processes.parquet` is still missing.",
        "",
        "## Plan A: Stable Automatic Line",
        "",
        "- Use `data/processed/pv_glass_cases.parquet`, `data/processed/naics_2017_enriched_corpus.parquet`, and `data/processed/epa_factors_v13.parquet`.",
        "- Keep process extension disabled in `configs/exp/full_pv_glass.yaml`.",
        "- Reuse existing local retriever and reranker model paths from `full.yaml`.",
        "",
        "Recommended commands:",
        "",
        "```bash",
    ]

    if not processed_present:
        lines.append("./.venv/bin/python scripts/01_prepare_main_data.py --amazon_dir data/raw/amazon_caml --epa_dir data/raw/epa_factors --naics_dir data/raw/naics --out_dir data/processed --config configs/exp/full.yaml")
    else:
        lines.append("# main processed benchmark assets already exist; no rebuild needed")

    if not splits_present:
        lines.append("./.venv/bin/python scripts/02_make_splits.py --input_path data/processed/products.parquet --out_dir data/splits --config configs/exp/full.yaml")
    else:
        lines.append("# benchmark splits already exist; no rebuild needed")

    lines.extend(
        [
            "./.venv/bin/python scripts/01e_build_pv_glass_cases.py",
            "./.venv/bin/python scripts/14_run_pv_glass_case_study.py --config configs/exp/full_pv_glass.yaml --mode smoke --seed 13",
            "./.venv/bin/python scripts/14_run_pv_glass_case_study.py --config configs/exp/full_pv_glass.yaml --mode predict --seed 13",
            "```",
            "",
            "## Plan B: Enhanced Process Retrieval",
            "",
            "- Wait until `data/processed/uslci_processes.parquet` or another orchestration-compatible process asset is available.",
            "- Keep the current `data/interim/pv_glass_process_corpus_standardized.csv` as a side corpus for standalone process retrieval, not as a replacement for the benchmark EPA target table.",
            "",
            "Recommended commands after process assets are ready:",
            "",
            "```bash",
        ]
    )
    if uslci_present:
        lines.append("./.venv/bin/python scripts/11_run_process_extension.py --products_path data/processed/pv_glass_cases.parquet --uslci_path data/processed/uslci_processes.parquet --prefilter_by_naics false --retriever_ckpt bm25 --output_dir reports/case_study/pv_glass/process_uslci")
    else:
        lines.append("# `data/processed/uslci_processes.parquet` is still missing; keep using the standalone process corpus path below for exploratory retrieval")
    lines.extend(
        [
            "./.venv/bin/python scripts/11_run_process_extension.py --products_path data/processed/pv_glass_cases.parquet --uslci_path data/interim/pv_glass_process_corpus_standardized.csv --prefilter_by_naics false --retriever_ckpt bm25 --output_dir reports/case_study/pv_glass/process_sidecar",
            "```",
        ]
    )

    out_path = root / "reports" / "audit" / "stage_c_run_plan.md"
    ensure_directory(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_pv_glass_case_study(
    *,
    config_path: str | Path,
    mode: str,
    seed: int,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    root = _repo_root(repo_root)
    config_path_obj = Path(config_path)
    if not config_path_obj.is_absolute():
        config_path_obj = (root / config_path_obj).resolve()
    config = load_yaml(config_path_obj)
    products_path = root / str(config.get("pv_cases_path") or config.get("case_study_products_path") or "data/processed/pv_glass_cases.parquet")
    corpus_path = root / str(config.get("corpus_path", "data/processed/naics_2017_enriched_corpus.parquet"))
    process_corpus_path = root / str(config.get("glass_process_corpus_path") or config.get("process_corpus_path") or "data/interim/pv_glass_process_corpus_standardized.csv")
    output_root = root / "reports" / "case_study" / "pv_glass"
    bm25_dir = output_root / "bm25"
    predict_dir = output_root / "predict"
    process_dir = output_root / "process_sidecar"

    commands: list[list[str]] = []
    if mode in {"smoke", "predict"}:
        commands.append(
            [
                sys.executable,
                "scripts/03_train_baselines.py",
                "--train_path",
                str(products_path.relative_to(root)),
                "--dev_path",
                str(products_path.relative_to(root)),
                "--corpus_path",
                str(corpus_path.relative_to(root)),
                "--model",
                "bm25",
                "--config",
                str(config_path_obj.relative_to(root)),
                "--output_dir",
                str(bm25_dir.relative_to(root)),
                "--seed",
                str(seed),
            ]
        )
    if mode in {"predict"}:
        commands.append(
            [
                sys.executable,
                "scripts/08_predict_all.py",
                "--split_path",
                str(products_path.relative_to(root)),
                "--corpus_path",
                str(corpus_path.relative_to(root)),
                "--retriever_ckpt",
                str((bm25_dir / "retrieval_topk_dev_bm25.jsonl").relative_to(root)),
                "--config",
                str(config_path_obj.relative_to(root)),
                "--output_dir",
                str(predict_dir.relative_to(root)),
            ]
        )
    if mode in {"process"}:
        commands.append(
            [
                sys.executable,
                "scripts/11_run_process_extension.py",
                "--products_path",
                str(products_path.relative_to(root)),
                "--uslci_path",
                str(process_corpus_path.relative_to(root)),
                "--prefilter_by_naics",
                "false",
                "--retriever_ckpt",
                "bm25",
                "--output_dir",
                str(process_dir.relative_to(root)),
            ]
        )

    results: dict[str, Path] = {}
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    if bm25_dir.exists():
        results["bm25_dir"] = bm25_dir
    if predict_dir.exists():
        results["predict_dir"] = predict_dir
    if process_dir.exists():
        results["process_dir"] = process_dir
    return results


def _inspect_csv_asset(path: Path, required_columns: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "readable": False,
            "row_count": 0,
            "columns": [],
            "missing_columns": required_columns,
            "all_empty_columns": [],
        }
    try:
        frame = _read_csv(path)
    except Exception as exc:
        return {
            "exists": True,
            "size_bytes": path.stat().st_size,
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "row_count": 0,
            "columns": [],
            "missing_columns": required_columns,
            "all_empty_columns": [],
        }
    missing_columns = [column for column in required_columns if column not in frame.columns]
    all_empty_columns = [
        column
        for column in frame.columns
        if frame[column].astype(str).str.strip().eq("").all()
    ]
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "readable": True,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "missing_columns": missing_columns,
        "all_empty_columns": all_empty_columns,
    }


def _inspect_parquet_asset(path: Path, required_columns: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "readable": False,
            "row_count": 0,
            "columns": [],
            "missing_columns": required_columns,
        }
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return {
            "exists": True,
            "size_bytes": path.stat().st_size,
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "row_count": 0,
            "columns": [],
            "missing_columns": required_columns,
        }
    missing_columns = [column for column in required_columns if column not in frame.columns]
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "readable": True,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "missing_columns": missing_columns,
    }


def inspect_stage_c_execution(repo_root: Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo_root)
    config_path = root / "configs" / "exp" / "full_pv_glass.yaml"
    config_exists = config_path.exists()
    config: dict[str, Any] = {}
    config_error = ""
    if config_exists:
        try:
            config = load_yaml(config_path)
        except Exception as exc:
            config_error = f"{type(exc).__name__}: {exc}"

    config_info = {
        "path": config_path,
        "exists": config_exists,
        "size_bytes": config_path.stat().st_size if config_exists else 0,
        "parsable": bool(config) and not config_error,
        "error": config_error,
        "whether_process_extension": config.get("whether_process_extension"),
        "uslci_path": config.get("uslci_path"),
        "retriever_path": config.get("dense_encoder_name") or config.get("model_name"),
        "reranker_path": config.get("rerank_base_model"),
        "retriever_exists": bool(config.get("dense_encoder_name") or config.get("model_name")) and (root / str(config.get("dense_encoder_name") or config.get("model_name"))).exists(),
        "reranker_exists": bool(config.get("rerank_base_model")) and (root / str(config.get("rerank_base_model"))).exists(),
        "pv_cases_path": config.get("pv_cases_path") or config.get("case_study_products_path"),
        "corpus_path": config.get("corpus_path"),
        "epa_factors_path": config.get("epa_factors_path"),
        "glass_registry_path": config.get("glass_registry_path"),
        "glass_process_corpus_path": config.get("glass_process_corpus_path"),
    }

    input_assets = {
        "data/raw/amazon_caml/pv_glass_cases.csv": _inspect_csv_asset(
            root / "data" / "raw" / "amazon_caml" / "pv_glass_cases.csv",
            ["product_id", "title", "description", "gold_naics_code"],
        ),
        "data/interim/pv_glass_cases_with_metadata.csv": _inspect_csv_asset(
            root / "data" / "interim" / "pv_glass_cases_with_metadata.csv",
            ["product_id", "title", "description", "gold_naics_code", "source_note", "source_file", "geography", "thickness_mm", "stage_hint"],
        ),
        "data/raw/epa_factors/epa_naics_v13.csv": _inspect_csv_asset(
            root / "data" / "raw" / "epa_factors" / "epa_naics_v13.csv",
            RAW_EPA_STANDARD_COLUMNS,
        ),
        "data/raw/naics/naics_2017_enriched.csv": _inspect_csv_asset(
            root / "data" / "raw" / "naics" / "naics_2017_enriched.csv",
            RAW_NAICS_STANDARD_COLUMNS,
        ),
        "data/interim/glass_factor_registry_standardized.csv": _inspect_csv_asset(
            root / "data" / "interim" / "glass_factor_registry_standardized.csv",
            GLASS_REGISTRY_REQUIRED_COLUMNS,
        ),
        "data/interim/pv_glass_process_corpus_standardized.csv": _inspect_csv_asset(
            root / "data" / "interim" / "pv_glass_process_corpus_standardized.csv",
            PROCESS_CORPUS_REQUIRED_COLUMNS,
        ),
    }

    processed_assets = {
        "data/processed/pv_glass_cases.parquet": _inspect_parquet_asset(
            root / "data" / "processed" / "pv_glass_cases.parquet",
            ["product_id", "title", "description", "text", "gold_naics_code"],
        ),
        "data/processed/epa_factors_v13.parquet": _inspect_parquet_asset(
            root / "data" / "processed" / "epa_factors_v13.parquet",
            RAW_EPA_STANDARD_COLUMNS,
        ),
        "data/processed/naics_2017_enriched_corpus.parquet": _inspect_parquet_asset(
            root / "data" / "processed" / "naics_2017_enriched_corpus.parquet",
            ["naics_code", "naics_text", "naics_title", "naics_code_6"],
        ),
    }

    plan_vs_actual = [
        {
            "planned_name": "data/processed/pv_glass_cases.parquet",
            "actual_runtime_key": "pv_cases_path / case_study_products_path",
            "actual_runtime_value": str(config_info["pv_cases_path"] or ""),
            "matches": str(config_info["pv_cases_path"] or "") == "data/processed/pv_glass_cases.parquet",
        },
        {
            "planned_name": "data/processed/naics_2017_enriched_corpus.parquet",
            "actual_runtime_key": "corpus_path",
            "actual_runtime_value": str(config_info["corpus_path"] or ""),
            "matches": str(config_info["corpus_path"] or "") == "data/processed/naics_2017_enriched_corpus.parquet",
        },
        {
            "planned_name": "data/processed/epa_factors_v13.parquet",
            "actual_runtime_key": "epa_factors_path",
            "actual_runtime_value": str(config_info["epa_factors_path"] or ""),
            "matches": str(config_info["epa_factors_path"] or "") == "data/processed/epa_factors_v13.parquet",
        },
    ]

    script_info = {
        "scripts/01e_build_pv_glass_cases.py": {
            "exists": (root / "scripts" / "01e_build_pv_glass_cases.py").exists(),
            "help_command": "./.venv/bin/python scripts/01e_build_pv_glass_cases.py --help",
            "inputs": [
                "data/Glass_EPD",
                "data/interim/aux_documents.csv",
            ],
            "outputs": [
                "data/raw/amazon_caml/pv_glass_cases.csv",
                "data/interim/pv_glass_cases_with_metadata.csv",
                "reports/audit/stage_b_glass_extension_audit.md",
            ],
            "generates_processed_assets": False,
        },
        "scripts/14_run_pv_glass_case_study.py": {
            "exists": (root / "scripts" / "14_run_pv_glass_case_study.py").exists(),
            "help_command": "./.venv/bin/python scripts/14_run_pv_glass_case_study.py --help",
            "inputs": [
                "configs/exp/full_pv_glass.yaml",
                "data/processed/pv_glass_cases.parquet",
                "data/processed/naics_2017_enriched_corpus.parquet",
                "data/processed/epa_factors_v13.parquet",
            ],
            "outputs": [
                "reports/case_study/pv_glass/bm25/retrieval_topk_dev_bm25.jsonl",
                "reports/case_study/pv_glass/predict/regression_preds_pv_glass_cases_top1_factor_lookup.parquet",
                "reports/case_study/pv_glass/predict/regression_preds_pv_glass_cases_topk_factor_mixture.parquet",
            ],
            "generates_processed_assets": False,
            "notes": [
                "`--mode smoke` only runs `scripts/03_train_baselines.py`; it is not a reduced-size smoke subset.",
                "`--mode predict` reruns the BM25 baseline step before `scripts/08_predict_all.py`.",
                "`glass_registry_path` is stored in config but is not consumed by `run_pv_glass_case_study()`.",
            ],
        },
        "scripts/01b_prepare_pv_glass_extension.py": {
            "exists": (root / "scripts" / "01b_prepare_pv_glass_extension.py").exists(),
            "help_command": "./.venv/bin/python scripts/01b_prepare_pv_glass_extension.py --help",
            "inputs": [
                "data/raw/amazon_caml/pv_glass_cases.csv or its source materials",
                "data/raw/naics/naics_2017_enriched.csv",
                "data/raw/epa_factors/epa_naics_v13.csv",
            ],
            "outputs": [
                "data/processed/pv_glass_cases.parquet",
                "data/processed/naics_2017_enriched_corpus.parquet",
                "data/processed/epa_factors_v13.parquet",
            ],
            "generates_processed_assets": True,
        },
    }

    required_for_plan_a = [
        config_info["parsable"],
        config_info["retriever_exists"],
        config_info["reranker_exists"],
        config_info["whether_process_extension"] is False,
        not config_info["uslci_path"],
        all(asset["exists"] and asset["readable"] and not asset["missing_columns"] for asset in input_assets.values()),
        all(asset["exists"] and asset["readable"] and not asset["missing_columns"] for asset in processed_assets.values()),
    ]
    closure = all(required_for_plan_a)

    return {
        "config": config_info,
        "input_assets": input_assets,
        "processed_assets": processed_assets,
        "plan_vs_actual": plan_vs_actual,
        "script_info": script_info,
        "closure": closure,
    }


def write_stage_c_execution_check(repo_root: Path | None = None) -> Path:
    root = _repo_root(repo_root)
    inspection = inspect_stage_c_execution(root)
    config = inspection["config"]

    lines = [
        "# Stage C Execution Check",
        "",
        "## Closure Verdict",
        "",
        f"- Stage C Plan A closed in the current workspace: `{inspection['closure']}`.",
        "- This verdict only covers the non-process-extension path based on `configs/exp/full_pv_glass.yaml`.",
        "",
        "## Config Check",
        "",
        f"- Config path: `{Path(config['path']).relative_to(root)}`.",
        f"- Exists and non-empty: `{config['exists'] and config['size_bytes'] > 0}`.",
        f"- YAML parses successfully: `{config['parsable']}`.",
        f"- Retriever path exists: `{config['retriever_exists']}` -> `{config['retriever_path']}`.",
        f"- Reranker path exists: `{config['reranker_exists']}` -> `{config['reranker_path']}`.",
        f"- `whether_process_extension`: `{config['whether_process_extension']}`.",
        f"- Residual `uslci_path`: `{config['uslci_path']}`.",
        "",
        "## Case-Study Input Assets",
        "",
        "| Asset | Exists | Readable | Rows | Missing required columns | All-empty columns |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rel, info in inspection["input_assets"].items():
        lines.append(
            f"| `{rel}` | `{info['exists']}` | `{info['readable']}` | `{info['row_count']}` | "
            f"{', '.join(f'`{col}`' for col in info['missing_columns']) if info['missing_columns'] else '(none)'} | "
            f"{', '.join(f'`{col}`' for col in info['all_empty_columns']) if info['all_empty_columns'] else '(none)'} |"
        )

    lines.extend(
        [
            "",
            "## Processed Assets",
            "",
            "| Planned name | Actual runtime key | Actual runtime value | Exists | Readable | Missing required columns | Match |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in inspection["plan_vs_actual"]:
        asset_info = inspection["processed_assets"][item["planned_name"]]
        lines.append(
            f"| `{item['planned_name']}` | `{item['actual_runtime_key']}` | `{item['actual_runtime_value']}` | "
            f"`{asset_info['exists']}` | `{asset_info['readable']}` | "
            f"{', '.join(f'`{col}`' for col in asset_info['missing_columns']) if asset_info['missing_columns'] else '(none)'} | "
            f"`{item['matches']}` |"
        )

    lines.extend(
        [
            "",
            "## Script Entry Points",
            "",
            "| Script | Exists | Generates processed assets | Declared outputs |",
            "| --- | --- | --- | --- |",
        ]
    )
    for rel, info in inspection["script_info"].items():
        lines.append(
            f"| `{rel}` | `{info['exists']}` | `{info['generates_processed_assets']}` | "
            f"{'; '.join(f'`{item}`' for item in info['outputs'])} |"
        )

    lines.extend(
        [
            "",
            "## I/O Findings",
            "",
            "- `scripts/01e_build_pv_glass_cases.py` is readable and its argparse parses, but it only rebuilds raw/interim case CSVs and the Stage B report. It does not create the processed parquet assets that Stage C runtime uses.",
            "- `scripts/14_run_pv_glass_case_study.py` is readable and its argparse parses. In `smoke` mode it runs only `scripts/03_train_baselines.py`; in `predict` mode it reruns the BM25 baseline step and then calls `scripts/08_predict_all.py`.",
            "- `scripts/14_run_pv_glass_case_study.py` consumes processed parquets and the config; it does not bootstrap missing processed case-study assets.",
            "- `scripts/01b_prepare_pv_glass_extension.py` is the actual lightweight bridge for missing processed case-study assets.",
            "- `glass_registry_path` and `glass_process_corpus_path` are present in `configs/exp/full_pv_glass.yaml`, but the current smoke/predict path does not read `glass_registry_path`.",
            "",
            "## Remaining Gaps",
            "",
        ]
    )

    remaining = []
    for rel, info in inspection["input_assets"].items():
        if not info["exists"] or not info["readable"] or info["missing_columns"]:
            remaining.append(rel)
    for rel, info in inspection["processed_assets"].items():
        if not info["exists"] or not info["readable"] or info["missing_columns"]:
            remaining.append(rel)
    if not inspection["closure"]:
        remaining.append("Stage C non-process path is not fully closed.")
    lines.extend(f"- `{item}`" for item in remaining or ["No blocking missing artifact remains for Plan A in the current workspace."])

    out_path = root / "reports" / "audit" / "stage_c_execution_check.md"
    ensure_directory(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_stage_c_commands(repo_root: Path | None = None) -> Path:
    root = _repo_root(repo_root)
    inspection = inspect_stage_c_execution(root)
    processed_present = all(
        info["exists"] and info["readable"] and not info["missing_columns"]
        for info in inspection["processed_assets"].values()
    )
    retrieval_exists = (root / "reports" / "case_study" / "pv_glass" / "bm25" / "retrieval_topk_dev_bm25.jsonl").exists()

    lines = [
        "# Stage C Commands",
        "",
        "## Minimal Runnable Experiment Commands",
        "",
        "1. Prepare or verify case-study assets.",
        "Precondition: run only if any of `data/processed/pv_glass_cases.parquet`, `data/processed/epa_factors_v13.parquet`, or `data/processed/naics_2017_enriched_corpus.parquet` is missing or stale.",
        "```bash",
        "./.venv/bin/python scripts/01b_prepare_pv_glass_extension.py",
        "```",
        "",
        "2. Build or refresh only the raw/interim case tables.",
        "Precondition: optional; skip if `data/raw/amazon_caml/pv_glass_cases.csv` and `data/interim/pv_glass_cases_with_metadata.csv` already exist and are current.",
        "```bash",
        "./.venv/bin/python scripts/01e_build_pv_glass_cases.py",
        "```",
        "",
        "3. Run the Stage C smoke path.",
        f"Precondition: {'all required processed assets are already present in this workspace, so the prepare step above is currently skippable.' if processed_present else 'processed case-study assets must exist first.'}",
        "```bash",
        "./.venv/bin/python scripts/14_run_pv_glass_case_study.py --config configs/exp/full_pv_glass.yaml --mode smoke --seed 13",
        "```",
        "",
        "4. Run the seed=13 prediction path without repeating unnecessary setup.",
        f"Precondition: {'the retrieval artifact already exists, so you can call `08_predict_all.py` directly.' if retrieval_exists else 'run the smoke command first so `reports/case_study/pv_glass/bm25/retrieval_topk_dev_bm25.jsonl` exists.'}",
        "```bash",
        "./.venv/bin/python scripts/08_predict_all.py --split_path data/processed/pv_glass_cases.parquet --corpus_path data/processed/naics_2017_enriched_corpus.parquet --retriever_ckpt reports/case_study/pv_glass/bm25/retrieval_topk_dev_bm25.jsonl --config configs/exp/full_pv_glass.yaml --output_dir reports/case_study/pv_glass/predict",
        "```",
        "",
        "Alternative single-command prediction path:",
        "Precondition: none beyond the processed assets and config, but this reruns the BM25 baseline step.",
        "```bash",
        "./.venv/bin/python scripts/14_run_pv_glass_case_study.py --config configs/exp/full_pv_glass.yaml --mode predict --seed 13",
        "```",
        "",
        "## Diagnostic Commands",
        "",
        "1. Check config parsing and key runtime fields.",
        "```bash",
        "PYTHONPATH=src ./.venv/bin/python -c \"from open_match_lca.io_utils import load_yaml; c=load_yaml('configs/exp/full_pv_glass.yaml'); print({'whether_process_extension': c.get('whether_process_extension'), 'uslci_path': c.get('uslci_path'), 'pv_cases_path': c.get('pv_cases_path') or c.get('case_study_products_path'), 'corpus_path': c.get('corpus_path'), 'epa_factors_path': c.get('epa_factors_path')})\"",
        "```",
        "",
        "2. Check key input files and sizes.",
        "```bash",
        "for f in configs/exp/full_pv_glass.yaml data/raw/amazon_caml/pv_glass_cases.csv data/interim/pv_glass_cases_with_metadata.csv data/raw/epa_factors/epa_naics_v13.csv data/raw/naics/naics_2017_enriched.csv data/interim/glass_factor_registry_standardized.csv data/interim/pv_glass_process_corpus_standardized.csv data/processed/pv_glass_cases.parquet data/processed/epa_factors_v13.parquet data/processed/naics_2017_enriched_corpus.parquet; do [ -e \"$f\" ] && stat -f \"%N %z bytes\" \"$f\" || echo \"MISSING $f\"; done",
        "```",
        "",
        "3. Check script argparse parsing.",
        "```bash",
        "./.venv/bin/python scripts/01e_build_pv_glass_cases.py --help",
        "./.venv/bin/python scripts/14_run_pv_glass_case_study.py --help",
        "./.venv/bin/python scripts/01b_prepare_pv_glass_extension.py --help",
        "```",
        "",
        "4. If a run fails, inspect the most recent logs.",
        "```bash",
        "ls -lt reports/logs | head -20",
        "```",
    ]

    out_path = root / "reports" / "audit" / "stage_c_commands.md"
    ensure_directory(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
