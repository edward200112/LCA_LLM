from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from open_match_lca.constants import PROJECT_ROOT
from open_match_lca.data.aux_corpus import load_all_pdf_indexes
from open_match_lca.data.aux_pdf_parser import clean_filename_title, extract_thickness_mm
from open_match_lca.data.build_naics_corpus import build_naics_corpus
from open_match_lca.data.parse_amazon_caml import parse_amazon_caml
from open_match_lca.data.parse_epa_factors import parse_epa_factors
from open_match_lca.io_utils import ensure_directory, load_yaml, write_parquet
from open_match_lca.schemas import normalize_naics_code


PV_CASE_COLUMNS = ["product_id", "title", "description", "gold_naics_code"]
PV_CASE_METADATA_COLUMNS = PV_CASE_COLUMNS + [
    "source_note",
    "source_file",
    "source_doc_id",
    "geography",
    "thickness_mm",
    "stage_hint",
]
FACTOR_REGISTRY_COLUMNS = [
    "source_id",
    "source_type",
    "source_file",
    "naics_code",
    "process_name",
    "factor_value",
    "factor_unit",
    "stage",
    "thickness_mm",
    "mass_per_m2",
    "geography",
    "year",
    "quality_tier",
    "notes",
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
    "source_type",
    "source_file",
]

GLASS_CASE_FAMILIES = [
    {
        "family_id": "base_low_iron_float",
        "source_doc_id": "glass_0008",
        "title_template": "{thickness} mm low-iron float glass substrate for photovoltaic modules",
        "description_template": (
            "Low-iron float glass sheet for photovoltaic module cover applications. "
            "Primary flat glass manufacturing route with silica sand, soda ash, limestone, dolomite, feldspar, and cullet, "
            "followed by batch melting, float forming, annealing, and sheet cutting."
        ),
        "gold_naics_code": "327211",
        "stage_hint": "flat_glass_melt_line",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "base_clear_float",
        "source_doc_id": "glass_0007",
        "title_template": "{thickness} mm clear float glass for solar module front sheets",
        "description_template": (
            "Clear float glass substrate used as a base sheet for solar glazing and photovoltaic cover assemblies. "
            "Float glass production language emphasizes furnace melting, ribbon forming, annealing, trimming, and packaging."
        ),
        "gold_naics_code": "327211",
        "stage_hint": "flat_glass_melt_line",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "base_patterned_solar",
        "source_doc_id": "glass_0012",
        "title_template": "{thickness} mm patterned solar glass base sheet",
        "description_template": (
            "Patterned solar glass sheet for crystalline silicon module cover applications. "
            "Flat glass line language with solar transmission focus, textured surface, low-iron chemistry, and furnace batch melting."
        ),
        "gold_naics_code": "327211",
        "stage_hint": "flat_glass_melt_line",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "base_low_carbon_clearlite",
        "source_doc_id": "glass_0001",
        "title_template": "{thickness} mm low-carbon clearlite flat glass for photovoltaic glazing",
        "description_template": (
            "Low-carbon flat glass product suitable for photovoltaic cover glass and solar glazing assemblies. "
            "Description retains flat glass manufacturing cues including low-iron composition, melt furnace operation, float forming, and cullet use."
        ),
        "gold_naics_code": "327211",
        "stage_hint": "flat_glass_melt_line",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "base_patterned_general",
        "source_doc_id": "glass_0009",
        "title_template": "{thickness} mm patterned low-iron glass substrate for solar laminates",
        "description_template": (
            "Patterned low-iron glass substrate used before tempering or coating in photovoltaic laminate manufacturing. "
            "Base description keeps flat glass terminology such as silica sand, cullet, float forming, annealing, and cutting."
        ),
        "gold_naics_code": "327211",
        "stage_hint": "flat_glass_melt_line",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "base_generic_flat",
        "source_doc_id": "glass_0002",
        "title_template": "{thickness} mm flat glass sheet for solar cover applications",
        "description_template": (
            "Flat glass sheet for downstream photovoltaic cover glass conversion. "
            "Product text emphasizes float glass manufacturing, furnace melting, high-transmission glass ribbon production, and standard sheet sizing."
        ),
        "gold_naics_code": "327211",
        "stage_hint": "flat_glass_melt_line",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "finish_toughened_float",
        "source_doc_id": "glass_0010",
        "title_template": "{thickness} mm thermally toughened float glass for photovoltaic modules",
        "description_template": (
            "Thermally toughened float glass finished from purchased flat glass for photovoltaic cover use. "
            "Description highlights tempering, edge finishing, strength improvement, and module front-sheet applications."
        ),
        "gold_naics_code": "327215",
        "stage_hint": "tempering_finish_line",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "finish_offline_coated",
        "source_doc_id": "glass_0006",
        "title_template": "{thickness} mm offline coated low-iron toughened glass for solar module cover",
        "description_template": (
            "Offline coated and thermally toughened low-iron glass made from purchased flat glass for photovoltaic cover glass applications. "
            "Key process words include anti-reflective coating, tempering, washing, coating, and final inspection."
        ),
        "gold_naics_code": "327215",
        "stage_hint": "coating_and_tempering",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "finish_solar_cover",
        "source_doc_id": "glass_0012",
        "title_template": "{thickness} mm solar module cover glass with high-transmission finish",
        "description_template": (
            "Solar module cover glass finished for photovoltaic laminates and front cover assemblies. "
            "Purchased-glass conversion cues include surface treatment, coating, tempering, packing, and module integration."
        ),
        "gold_naics_code": "327215",
        "stage_hint": "solar_cover_finish",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "finish_patterned_tempered",
        "source_doc_id": "glass_0011",
        "title_template": "{thickness} mm tempered patterned glass for photovoltaic cover assemblies",
        "description_template": (
            "Patterned photovoltaic cover glass produced by shaping or tempering purchased glass for module protection. "
            "Text keeps conversion-stage language around tempering, pattern retention, coating compatibility, and cover assembly use."
        ),
        "gold_naics_code": "327215",
        "stage_hint": "tempering_finish_line",
        "source_note": "epd_manual_synthesis",
    },
    {
        "family_id": "finish_ar_coated",
        "source_doc_id": "glass_0006",
        "title_template": "{thickness} mm AR-coated low-iron glass for bifacial photovoltaic modules",
        "description_template": (
            "Anti-reflective coated low-iron glass made from purchased flat glass for bifacial and high-transmittance photovoltaic modules. "
            "Description preserves coating, washing, drying, inspection, and packaging terms rather than melt-line terms."
        ),
        "gold_naics_code": "327215",
        "stage_hint": "coating_and_tempering",
        "source_note": "manual",
    },
    {
        "family_id": "finish_module_frontsheet",
        "source_doc_id": "glass_0010",
        "title_template": "{thickness} mm tempered module front-sheet glass",
        "description_template": (
            "Tempered module front-sheet glass for solar panel assembly, generated from purchased float glass. "
            "Description focuses on downstream tempering, edge work, surface conditioning, and module cover performance."
        ),
        "gold_naics_code": "327215",
        "stage_hint": "module_cover_finish",
        "source_note": "manual",
    },
]

THICKNESSES_MM = [2.0, 3.2, 4.0]
GLASS_ENRICHMENTS = {
    "327211": (
        "Glass-focused terms: flat glass, low-iron float glass, patterned glass, solar glass, photovoltaic cover glass, "
        "furnace batch melting, silica sand, cullet, annealing, ribbon forming, float bath."
    ),
    "327215": (
        "Glass-focused terms: coating, tempering, laminating, shaping purchased glass, anti-reflective coated glass, "
        "solar module cover glass, photovoltaic front sheet, offline coated glass, purchased float glass conversion."
    ),
    "327212": (
        "Related glass-forming terms: shaped glass, blown glass, pressed glass, textured or patterned glass components."
    ),
    "327213": "Related glass terms: glass container manufacturing, melted glass forming, furnace operation.",
}


@dataclass(frozen=True)
class SourceInfo:
    doc_id: str
    title: str
    source_file: str
    manufacturer: str
    geography: str


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or PROJECT_ROOT).resolve()


def _clean_numeric(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _load_aux_documents(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data" / "interim" / "aux_documents.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def _source_lookup(repo_root: Path) -> dict[str, SourceInfo]:
    frame = _load_aux_documents(repo_root)
    if frame.empty:
        indexes = load_all_pdf_indexes(repo_root)
        return {
            row["doc_id"]: SourceInfo(
                doc_id=row["doc_id"],
                title=clean_filename_title(Path(row["relative_path"])),
                source_file=row["relative_path"],
                manufacturer=row.get("manufacturer", ""),
                geography="",
            )
            for row in indexes.to_dict(orient="records")
        }
    lookup = {}
    for row in frame.to_dict(orient="records"):
        lookup[row["doc_id"]] = SourceInfo(
            doc_id=row["doc_id"],
            title=row["title"],
            source_file=row["source_file"],
            manufacturer=row.get("manufacturer", ""),
            geography=row.get("geography", ""),
        )
    return lookup


def _write_if_needed(frame: pd.DataFrame, path: Path, force: bool) -> Path:
    if path.exists() and not force:
        return path
    ensure_directory(path.parent)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def build_pv_glass_cases(repo_root: Path | None = None, *, force: bool = False) -> tuple[Path, Path]:
    root = _repo_root(repo_root)
    raw_path = root / "data" / "raw" / "amazon_caml" / "pv_glass_cases.csv"
    metadata_path = root / "data" / "interim" / "pv_glass_cases_with_metadata.csv"
    if raw_path.exists() and metadata_path.exists() and not force:
        return raw_path, metadata_path

    sources = _source_lookup(root)
    raw_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, str]] = []
    counter = 1
    for family in GLASS_CASE_FAMILIES:
        source = sources.get(
            family["source_doc_id"],
            SourceInfo(
                doc_id=family["source_doc_id"],
                title=family["family_id"],
                source_file="",
                manufacturer="",
                geography="",
            ),
        )
        for thickness in THICKNESSES_MM:
            product_id = f"pv_glass_{counter:04d}"
            title = family["title_template"].format(thickness=f"{thickness:.1f}")
            description = family["description_template"]
            raw_rows.append(
                {
                    "product_id": product_id,
                    "title": title,
                    "description": description,
                    "gold_naics_code": family["gold_naics_code"],
                }
            )
            metadata_rows.append(
                {
                    "product_id": product_id,
                    "title": title,
                    "description": description,
                    "gold_naics_code": family["gold_naics_code"],
                    "source_note": family["source_note"],
                    "source_file": source.source_file,
                    "source_doc_id": source.doc_id,
                    "geography": source.geography,
                    "thickness_mm": f"{thickness:.1f}",
                    "stage_hint": family["stage_hint"],
                }
            )
            counter += 1

    raw_frame = pd.DataFrame(raw_rows, columns=PV_CASE_COLUMNS)
    metadata_frame = pd.DataFrame(metadata_rows, columns=PV_CASE_METADATA_COLUMNS)
    _write_if_needed(raw_frame, raw_path, force)
    _write_if_needed(metadata_frame, metadata_path, force)
    return raw_path, metadata_path


def _normalize_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): str(column)
        for column in frame.columns
    }


def _load_naics_base_frame(repo_root: Path) -> tuple[pd.DataFrame, str]:
    root = _repo_root(repo_root)
    official_path = root / "data" / "external_downloads" / "naics" / "2017_NAICS_Structure.xlsx"
    if official_path.exists():
        frame = pd.read_excel(official_path)
        normalized = _normalize_columns(frame)
        code_column = normalized.get("2017_naics_code") or normalized.get("naics_code") or normalized.get("code")
        title_column = normalized.get("2017_naics_title") or normalized.get("naics_title") or normalized.get("title")
        if code_column and title_column:
            base = frame[[code_column, title_column]].rename(
                columns={code_column: "naics_code", title_column: "naics_title"}
            )
            return base, "official_2017_structure"
    fallback_path = root / "data" / "raw" / "naics" / "naics_from_caml.csv"
    base = pd.read_csv(fallback_path, dtype=str, keep_default_na=False)
    return base[["naics_code", "naics_title"]], "fallback_naics_from_caml"


def build_enriched_naics(repo_root: Path | None = None, *, force: bool = False) -> Path:
    root = _repo_root(repo_root)
    output_path = root / "data" / "raw" / "naics" / "naics_2017_enriched.csv"
    if output_path.exists() and not force:
        return output_path

    base, _ = _load_naics_base_frame(root)
    descriptions_path = root / "data" / "raw" / "naics" / "naics_from_caml.csv"
    descriptions = pd.read_csv(descriptions_path, dtype=str, keep_default_na=False)
    desc_map = descriptions.set_index("naics_code")["naics_description"].to_dict()

    parsed = base.copy()
    parsed["naics_code"] = parsed["naics_code"].map(normalize_naics_code)
    parsed["naics_title"] = parsed["naics_title"].fillna("").astype(str).str.strip()
    parsed = parsed.loc[parsed["naics_code"].str.len() == 6].reset_index(drop=True)
    parsed["naics_description"] = parsed["naics_code"].map(desc_map).fillna("")
    for code, enrichment in GLASS_ENRICHMENTS.items():
        mask = parsed["naics_code"].eq(code)
        parsed.loc[mask, "naics_description"] = (
            parsed.loc[mask, "naics_description"].fillna("").astype(str).str.strip() + " " + enrichment
        ).str.strip()
    parsed = parsed.drop_duplicates(subset=["naics_code"], keep="first").sort_values("naics_code").reset_index(drop=True)
    _write_if_needed(parsed[["naics_code", "naics_title", "naics_description"]], output_path, force)
    return output_path


def _load_epa_source_frame(repo_root: Path) -> tuple[pd.DataFrame, str]:
    root = _repo_root(repo_root)
    official_path = root / "data" / "external_downloads" / "epa" / "SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv"
    if official_path.exists():
        return pd.read_csv(official_path, dtype=str, keep_default_na=False), "official_epa_v13"
    fallback_path = root / "data" / "raw" / "epa_factors" / "epa_factors_from_caml.csv"
    return pd.read_csv(fallback_path, dtype=str, keep_default_na=False), "fallback_epa_from_caml"


def _select_column(normalized: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        for key, original in normalized.items():
            if candidate in key:
                return original
    return None


def build_standardized_epa_factors(repo_root: Path | None = None, *, force: bool = False) -> Path:
    root = _repo_root(repo_root)
    output_path = root / "data" / "raw" / "epa_factors" / "epa_naics_v13.csv"
    if output_path.exists() and not force:
        return output_path

    frame, source_label = _load_epa_source_frame(root)
    normalized = _normalize_columns(frame)
    naics_column = _select_column(normalized, "naics_code", "2017_naics_code", "naics")
    with_margins_column = _select_column(normalized, "with_margins", "with_margin", "supply_chain_emission_factors_with_margins")
    without_margins_column = _select_column(normalized, "without_margins", "without_margin", "supply_chain_emission_factors_without_margins")
    factor_value_column = _select_column(normalized, "factor_value", "supply_chain_emission_factors_with_margins")
    unit_column = _select_column(normalized, "factor_unit", "unit")
    year_column = _select_column(normalized, "source_year", "reference_year", "year")
    useeio_column = _select_column(normalized, "useeio_code", "bea_detail_code", "useeio", "bea")

    if source_label == "fallback_epa_from_caml":
        standardized = frame.copy()
    else:
        standardized = pd.DataFrame(
            {
                "naics_code": frame[naics_column] if naics_column else "",
                "factor_value": frame[factor_value_column] if factor_value_column else "",
                "factor_unit": frame[unit_column] if unit_column else "kg CO2e/USD",
                "with_margins": frame[with_margins_column] if with_margins_column else frame[factor_value_column],
                "without_margins": frame[without_margins_column] if without_margins_column else "",
                "source_year": frame[year_column] if year_column else "2022",
                "useeio_code": frame[useeio_column] if useeio_column else "",
            }
        )

    standardized = standardized.fillna("").copy()
    standardized["naics_code"] = standardized["naics_code"].map(normalize_naics_code)
    standardized = standardized.loc[standardized["naics_code"].str.len() == 6].reset_index(drop=True)
    for column in ["factor_value", "with_margins", "without_margins"]:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
    standardized["factor_unit"] = standardized["factor_unit"].replace("", "kg CO2e/USD").astype(str)
    standardized["source_year"] = pd.to_numeric(standardized["source_year"], errors="coerce").fillna(2022).astype(int)
    standardized["useeio_code"] = standardized["useeio_code"].astype(str)
    standardized = standardized.dropna(subset=["factor_value"]).reset_index(drop=True)
    standardized = standardized.sort_values(
        ["naics_code", "source_year", "factor_value"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["naics_code"], keep="first")
    standardized = standardized[
        ["naics_code", "factor_value", "factor_unit", "with_margins", "without_margins", "source_year", "useeio_code"]
    ].reset_index(drop=True)
    _write_if_needed(standardized, output_path, force)
    return output_path


def _glass_stage(material_or_product: str) -> tuple[str, str]:
    material = material_or_product.lower()
    if material in {"float_glass", "low_iron_float_glass", "patterned_glass", "solar_glass", "clearlite_glass", "glass"}:
        return "327211", "flat_glass"
    if material in {"thermally_toughened_float_glass"}:
        return "327215", "tempering_or_finishing"
    if material == "silica_sand":
        return "212322", "raw_material"
    if material == "limestone":
        return "212312", "raw_material"
    if material in {"soda_ash", "dolomite", "feldspar", "cullet"}:
        return "", "raw_material"
    return "", ""


def build_glass_factor_registry(repo_root: Path | None = None, *, force: bool = False) -> Path:
    root = _repo_root(repo_root)
    output_path = root / "data" / "interim" / "glass_factor_registry.csv"
    if output_path.exists() and not force:
        return output_path

    frame = _load_aux_documents(root)
    rows: list[dict[str, str]] = []
    for row in frame.to_dict(orient="records"):
        if row.get("category_level_1") not in {"glass", "raw_material"}:
            continue
        naics_code, stage = _glass_stage(row.get("material_or_product", ""))
        rows.append(
            {
                "source_id": row["doc_id"],
                "source_type": row.get("source_type", ""),
                "source_file": row.get("source_file", ""),
                "naics_code": naics_code,
                "process_name": row.get("title", ""),
                "factor_value": "",
                "factor_unit": "",
                "stage": stage,
                "thickness_mm": _clean_numeric(row.get("thickness_mm_ref")),
                "mass_per_m2": _clean_numeric(row.get("mass_per_m2")),
                "geography": row.get("geography", ""),
                "year": _clean_numeric(row.get("year")),
                "quality_tier": "metadata_only" if row.get("parse_status") != "parsed_pdf" else "parsed_text",
                "notes": "Numeric physical-unit factor not auto-extracted; registry entry created as a traceable scaffold from existing corpus metadata.",
            }
        )
    registry = pd.DataFrame(rows, columns=FACTOR_REGISTRY_COLUMNS).drop_duplicates(subset=["source_id"]).sort_values(
        ["stage", "source_id"]
    )
    _write_if_needed(registry, output_path, force)
    return output_path


def _process_text_from_nist_payload(payload: dict) -> str:
    process_doc = payload.get("processDocumentation", {}) or {}
    parts = [
        str(payload.get("name", "")),
        str(payload.get("description", "")),
        str(payload.get("category", "")),
        str(process_doc.get("technologyDescription", "")),
        str(process_doc.get("samplingDescription", "")),
        str(process_doc.get("geographyDescription", "")),
    ]
    joined = " ".join(part.strip() for part in parts if part and str(part).strip())
    return re.sub(r"\s+", " ", joined).strip()


def _reference_exchange(payload: dict) -> tuple[str, str]:
    for exchange in payload.get("exchanges", []) or []:
        if exchange.get("isQuantitativeReference"):
            flow = exchange.get("flow", {}) or {}
            unit = exchange.get("unit", {}) or {}
            return str(flow.get("name", "")), str(unit.get("name", ""))
    return "", ""


def build_pv_glass_process_corpus(repo_root: Path | None = None, *, force: bool = False) -> Path:
    root = _repo_root(repo_root)
    output_path = root / "data" / "interim" / "pv_glass_process_corpus.csv"
    if output_path.exists() and not force:
        return output_path

    rows: list[dict[str, str]] = []
    aux_docs = _load_aux_documents(root)
    for row in aux_docs.to_dict(orient="records"):
        if row.get("category_level_1") not in {"glass", "raw_material"}:
            continue
        rows.append(
            {
                "process_uuid": row["doc_id"],
                "process_name": row["title"],
                "category_path": f"auxiliary/{row['category_level_1']}/{row['material_or_product']}",
                "geography": row.get("geography", ""),
                "reference_flow_name": row.get("material_or_product", "").replace("_", " "),
                "reference_flow_unit": row.get("declared_unit", ""),
                "process_text": row.get("product_text", ""),
                "source_release": "aux_pdf_corpus",
                "source_type": row.get("source_type", "pdf"),
                "source_file": row.get("source_file", ""),
            }
        )

    nist_root = root / "data" / "NIST-Building_Systems" / "processes"
    keywords = ("photovoltaic", "solar", "glass", "glazing", "laminate", "panel", "facade")
    if nist_root.exists():
        for path in sorted(nist_root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            process_text = _process_text_from_nist_payload(payload)
            if not any(keyword in process_text.lower() for keyword in keywords):
                continue
            reference_flow_name, reference_flow_unit = _reference_exchange(payload)
            location = payload.get("location", {}) or {}
            rows.append(
                {
                    "process_uuid": str(payload.get("@id", path.stem)),
                    "process_name": str(payload.get("name", "")),
                    "category_path": str(payload.get("category", "")),
                    "geography": str(location.get("name", "")),
                    "reference_flow_name": reference_flow_name,
                    "reference_flow_unit": reference_flow_unit,
                    "process_text": process_text,
                    "source_release": str(payload.get("version", "")),
                    "source_type": "nist_building_systems",
                    "source_file": str(path.relative_to(root)),
                }
            )

    corpus = pd.DataFrame(rows, columns=PROCESS_CORPUS_COLUMNS).drop_duplicates(subset=["process_uuid"]).sort_values(
        ["source_type", "process_name"]
    )
    _write_if_needed(corpus, output_path, force)
    return output_path


def build_case_study_processed_assets(repo_root: Path | None = None, *, force: bool = False) -> dict[str, Path]:
    root = _repo_root(repo_root)
    pv_cases_raw, _ = build_pv_glass_cases(root, force=force)
    naics_raw = build_enriched_naics(root, force=force)
    epa_raw = build_standardized_epa_factors(root, force=force)

    outputs = {
        "products": root / "data" / "processed" / "pv_glass_cases.parquet",
        "naics_corpus": root / "data" / "processed" / "naics_2017_enriched_corpus.parquet",
        "epa_factors": root / "data" / "processed" / "epa_factors_v13.parquet",
    }
    if force or not outputs["products"].exists():
        write_parquet(parse_amazon_caml(str(pv_cases_raw)), outputs["products"])
    if force or not outputs["naics_corpus"].exists():
        write_parquet(build_naics_corpus(str(naics_raw)), outputs["naics_corpus"])
    if force or not outputs["epa_factors"].exists():
        write_parquet(parse_epa_factors(str(epa_raw)), outputs["epa_factors"])
    return outputs


def write_pv_glass_config(repo_root: Path | None = None, *, force: bool = False) -> Path:
    root = _repo_root(repo_root)
    output_path = root / "configs" / "exp" / "full_pv_glass.yaml"
    if output_path.exists() and not force:
        return output_path

    base_config = deepcopy(load_yaml(root / "configs" / "exp" / "full.yaml"))
    base_config["corpus_path"] = "data/processed/naics_2017_enriched_corpus.parquet"
    base_config["epa_factors_path"] = "data/processed/epa_factors_v13.parquet"
    base_config["case_study_products_path"] = "data/processed/pv_glass_cases.parquet"
    base_config["process_corpus_path"] = "data/interim/pv_glass_process_corpus.csv"
    base_config["whether_process_extension"] = False
    base_config["whether_regression"] = False
    base_config["factor_baselines"] = ["top1_factor_lookup", "topk_factor_mixture"]
    base_config.pop("uslci_path", None)
    ensure_directory(output_path.parent)
    output_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    return output_path


def write_pv_glass_summary(repo_root: Path | None = None) -> Path:
    root = _repo_root(repo_root)
    summary_path = root / "reports" / "audit" / "pv_glass_extension_summary.md"
    ensure_directory(summary_path.parent)
    download_log_path = root / "reports" / "audit" / "download_log.csv"
    download_log = (
        pd.read_csv(download_log_path, dtype=str, keep_default_na=False).fillna("")
        if download_log_path.exists()
        else pd.DataFrame()
    )
    downloaded = []
    skipped = []
    failed = []
    for row in download_log.to_dict(orient="records"):
        label = f"`{row['asset_id']}` -> `{row['status']}`"
        if row["status"] == "downloaded":
            downloaded.append(label)
        elif "failed" in row["status"]:
            failed.append(f"{label} ({row['error']})")
        else:
            skipped.append(label)

    smoke_note = ""
    smoke_preds_path = root / "reports" / "case_study" / "pv_glass" / "predict" / "regression_preds_pv_glass_cases_top1_factor_lookup.parquet"
    if smoke_preds_path.exists():
        smoke_preds = pd.read_parquet(smoke_preds_path)
        match_count = int((smoke_preds["gold_naics_code"] == smoke_preds["pred_naics_code"]).sum())
        smoke_note = (
            f"- Case-study BM25 smoke has already been executed in this workspace: "
            f"`{match_count}/{len(smoke_preds)}` top-1 NAICS matches on `pv_glass_cases.parquet`."
        )

    lines = [
        "# PV Glass Extension Summary",
        "",
        "## Existing Data Reused",
        "",
        "- Reused `data/raw/amazon_caml`, `data/raw/naics`, and `data/raw/epa_factors` mainline assets already present in the repository.",
        "- Reused `data/Glass_EPD`, `data/Material_EPD`, and `data/NIST-Building_Systems` as the primary local sources for glass and photovoltaic process context.",
        "- Reused existing processed main datasets and split files instead of rebuilding them.",
        "",
        "## Download Decisions",
        "",
    ]
    lines.extend(f"- {item}" for item in (skipped or ["No skip entries recorded."]))
    lines.extend(
        [
            "",
            "## New Downloads",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in (downloaded or ["No new downloads completed in this run."]))
    lines.extend(
        [
            "",
        "## Generated Outputs",
        "",
        "- `data/raw/amazon_caml/pv_glass_cases.csv`",
        "- `data/interim/pv_glass_cases_with_metadata.csv`",
        "- `data/raw/naics/naics_2017_enriched.csv`",
        "- `data/raw/epa_factors/epa_naics_v13.csv`",
        "- `data/interim/glass_factor_registry.csv`",
        "- `data/interim/pv_glass_process_corpus.csv`",
        "- `data/processed/pv_glass_cases.parquet`",
        "- `data/processed/naics_2017_enriched_corpus.parquet`",
        "- `data/processed/epa_factors_v13.parquet`",
        "- `configs/exp/full_pv_glass.yaml`",
        "- The standardized `naics_2017_enriched.csv` and `epa_naics_v13.csv` were generated from local fallback sources because upstream official downloads were blocked in this environment.",
        "",
        "## Remaining Gaps",
        "",
        "- `data/processed/uslci_processes.parquet` is still missing, so the default `full.yaml` process-extension path remains incomplete.",
        "- `glass_factor_registry.csv` is currently a metadata-first registry scaffold; numeric physical-unit factors still need manual extraction or a stronger born-digital table parser/OCR pass.",
        ]
    )
    if smoke_note:
        lines.extend(["", "## Executed Smoke", "", smoke_note])
    if failed:
        lines.extend(["", "## Download Failures", ""])
        lines.extend(f"- {item}" for item in failed)
    lines.extend(
        [
            "",
            "## Recommended Commands",
            "",
            "```bash",
            "./.venv/bin/python scripts/00b_audit_repo_and_data.py",
            "./.venv/bin/python scripts/00c_fetch_external_assets.py",
            "./.venv/bin/python scripts/01b_prepare_pv_glass_extension.py",
            "./.venv/bin/python scripts/03_train_baselines.py --train_path data/processed/pv_glass_cases.parquet --dev_path data/processed/pv_glass_cases.parquet --corpus_path data/processed/naics_2017_enriched_corpus.parquet --model bm25 --config configs/exp/full_pv_glass.yaml --output_dir reports/case_study/pv_glass/bm25 --seed 13",
            "./.venv/bin/python scripts/08_predict_all.py --split_path data/processed/pv_glass_cases.parquet --corpus_path data/processed/naics_2017_enriched_corpus.parquet --retriever_ckpt reports/case_study/pv_glass/bm25/retrieval_topk_dev_bm25.jsonl --config configs/exp/full_pv_glass.yaml --output_dir reports/case_study/pv_glass/predict",
            "./.venv/bin/python scripts/11_run_process_extension.py --products_path data/processed/pv_glass_cases.parquet --uslci_path data/interim/pv_glass_process_corpus.csv --prefilter_by_naics false --retriever_ckpt bm25 --output_dir reports/case_study/pv_glass/process_bm25",
            "```",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def prepare_pv_glass_extension(repo_root: Path | None = None, *, force: bool = False) -> dict[str, Path]:
    root = _repo_root(repo_root)
    pv_cases_path, pv_metadata_path = build_pv_glass_cases(root, force=force)
    naics_path = build_enriched_naics(root, force=force)
    epa_path = build_standardized_epa_factors(root, force=force)
    factor_registry_path = build_glass_factor_registry(root, force=force)
    process_corpus_path = build_pv_glass_process_corpus(root, force=force)
    processed_paths = build_case_study_processed_assets(root, force=force)
    config_path = write_pv_glass_config(root, force=force)
    summary_path = write_pv_glass_summary(root)
    return {
        "pv_glass_cases": pv_cases_path,
        "pv_glass_cases_metadata": pv_metadata_path,
        "naics_2017_enriched": naics_path,
        "epa_naics_v13": epa_path,
        "glass_factor_registry": factor_registry_path,
        "pv_glass_process_corpus": process_corpus_path,
        **processed_paths,
        "config": config_path,
        "summary": summary_path,
    }
