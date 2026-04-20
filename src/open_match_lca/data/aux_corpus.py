from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from open_match_lca.data.aux_pdf_parser import (
    build_product_text,
    candidate_description_lines,
    clean_filename_title,
    extract_declared_unit,
    extract_geography,
    extract_mass_per_m2,
    extract_pdf_text,
    extract_system_boundary,
    extract_thickness_mm,
    extract_year,
)
from open_match_lca.data.ocr_adapter import extract_text_with_ocr
from open_match_lca.io_utils import ensure_directory

INDEX_COLUMNS = [
    "doc_id",
    "file_name",
    "relative_path",
    "doc_type",
    "category_level_1",
    "material_or_product",
    "manufacturer",
    "source_url",
    "notes",
]

AUX_DOCUMENT_COLUMNS = [
    "doc_id",
    "title",
    "product_text",
    "category_level_1",
    "material_or_product",
    "manufacturer",
    "source_type",
    "source_file",
    "declared_unit",
    "thickness_mm_ref",
    "mass_per_m2",
    "system_boundary",
    "geography",
    "year",
    "raw_text_path",
    "parse_status",
]

LABEL_TEMPLATE_COLUMNS = [
    "sample_id",
    "doc_id",
    "product_text",
    "category_level_1",
    "material_or_product",
    "proposed_naics_code",
    "gold_naics_code",
    "label_status",
    "comments",
]


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    root_dir: str
    index_name: str
    doc_prefix: str
    category_level_1: str


AUX_CORPORA = [
    CorpusSpec(
        name="glass",
        root_dir="Glass_EPD",
        index_name="index.csv",
        doc_prefix="glass",
        category_level_1="glass",
    ),
    CorpusSpec(
        name="material",
        root_dir="Material_EPD",
        index_name="index.csv",
        doc_prefix="mat",
        category_level_1="raw_material",
    ),
    CorpusSpec(
        name="coal",
        root_dir="Coal_EPD",
        index_name="index.csv",
        doc_prefix="coal",
        category_level_1="fuel",
    ),
]

MATERIAL_SUBDIRS = [
    "silica_sand",
    "limestone",
    "soda_ash",
    "dolomite",
    "feldspar",
    "cullet",
    "unknown",
]

COAL_SUBDIRS = [
    "coal_quality",
    "coal_specs",
    "coal_background",
    "unknown",
]

MATERIAL_TOP_LEVEL_TARGETS = {
    "EPD document EPD-IES-0024953_003 en.pdf": "limestone",
    "EPD document S-P-12716 en.pdf": "silica_sand",
    "EPD document_EPD-IES-0030293_001_en.pdf": "silica_sand",
    "EPD document_EPD-IES-0025910_004_en.pdf": "unknown",
    "EPD document_EPD-IES-0030299_001_en.pdf": "unknown",
}

COAL_TOP_LEVEL_TARGETS = {
    "QGESSDetailedCoalSpecifications_010112.pdf": "coal_specs",
    "ds975.pdf": "unknown",
}

GLASS_MANUFACTURERS = {
    "agc": "AGC",
    "pilkington": "Pilkington",
    "sisecam": "Sisecam",
}

GLASS_PRODUCT_PATTERNS = [
    ("solar", "solar_glass"),
    ("patterned", "patterned_glass"),
    ("low-iron", "low_iron_float_glass"),
    ("low iron", "low_iron_float_glass"),
    ("toughened", "thermally_toughened_float_glass"),
    ("float", "float_glass"),
    ("clearlite", "clearlite_glass"),
]

DOC_TYPE_PATTERNS = [
    ("epd", "epd"),
    ("datasheet", "tds"),
    ("tds", "tds"),
    ("specification", "spec"),
    ("spec", "spec"),
    ("standard", "standard"),
    ("is.", "standard"),
    ("guideline", "report"),
    ("report", "report"),
]

TYPE_LABELS = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".parquet": "parquet",
    ".pdf": "pdf",
    ".pkl": "pkl",
    ".txt": "txt",
}


def _repo_data_dir(repo_root: Path) -> Path:
    return repo_root / "data"


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _empty_dir_gitkeep(path: Path) -> None:
    has_files = any(child.is_file() and child.name != ".gitkeep" for child in path.iterdir())
    if has_files:
        gitkeep = path / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()
        return
    gitkeep = path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")


def _safe_move(source: Path, destination: Path) -> Path:
    ensure_directory(destination.parent)
    if not destination.exists():
        source.rename(destination)
        return destination

    if _sha1(source) == _sha1(destination):
        renamed = destination.with_name(f"{destination.stem} duplicate top-level{destination.suffix}")
        if not renamed.exists():
            source.rename(renamed)
        else:
            counter = 2
            while True:
                candidate = destination.with_name(f"{destination.stem} duplicate top-level {counter}{destination.suffix}")
                if not candidate.exists():
                    source.rename(candidate)
                    renamed = candidate
                    break
                counter += 1
        return renamed

    counter = 2
    while True:
        candidate = destination.with_name(f"{destination.stem} moved {counter}{destination.suffix}")
        if not candidate.exists():
            source.rename(candidate)
            return candidate
        counter += 1


def normalize_aux_directories(repo_root: Path) -> list[str]:
    data_dir = _repo_data_dir(repo_root)
    operations: list[str] = []

    material_root = data_dir / "Material_EPD"
    for subdir in MATERIAL_SUBDIRS:
        ensure_directory(material_root / subdir)

    for pdf_path in sorted(material_root.glob("*.pdf")):
        target_subdir = MATERIAL_TOP_LEVEL_TARGETS.get(pdf_path.name, "unknown")
        new_path = _safe_move(pdf_path, material_root / target_subdir / pdf_path.name)
        operations.append(f"moved {pdf_path.relative_to(repo_root)} -> {new_path.relative_to(repo_root)}")

    for subdir in MATERIAL_SUBDIRS:
        _empty_dir_gitkeep(material_root / subdir)

    coal_root = data_dir / "Coal_EPD"
    for subdir in COAL_SUBDIRS:
        ensure_directory(coal_root / subdir)

    for pdf_path in sorted(coal_root.glob("*.pdf")):
        target_subdir = COAL_TOP_LEVEL_TARGETS.get(pdf_path.name, "unknown")
        new_path = _safe_move(pdf_path, coal_root / target_subdir / pdf_path.name)
        operations.append(f"moved {pdf_path.relative_to(repo_root)} -> {new_path.relative_to(repo_root)}")

    for subdir in COAL_SUBDIRS:
        _empty_dir_gitkeep(coal_root / subdir)

    return operations


def _suffix_counts(directory: Path) -> str:
    counts: dict[str, int] = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        label = TYPE_LABELS.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "other")
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "empty"
    return ", ".join(f"{label} x{counts[label]}" for label in sorted(counts))


def audit_data_directories(repo_root: Path) -> dict[str, dict[str, str]]:
    data_dir = _repo_data_dir(repo_root)
    audit_targets = {
        "data/raw": {
            "path": data_dir / "raw",
            "role": "main_flow",
            "purpose": "Raw source tables for the main product->NAICS->EPA/USEEIO task.",
        },
        "data/interim": {
            "path": data_dir / "interim",
            "role": "main_flow_and_aux",
            "purpose": "Generated intermediate artifacts, including the new auxiliary corpus tables.",
        },
        "data/processed": {
            "path": data_dir / "processed",
            "role": "main_flow",
            "purpose": "Prepared tabular datasets for training and evaluation.",
        },
        "data/splits": {
            "path": data_dir / "splits",
            "role": "main_flow",
            "purpose": "Train/dev/test splits for the main benchmark.",
        },
        "data/Glass_EPD": {
            "path": data_dir / "Glass_EPD",
            "role": "auxiliary_pdf_corpus",
            "purpose": "Glass product EPD/support PDFs used as retrieval and labeling support material.",
        },
        "data/Material_EPD": {
            "path": data_dir / "Material_EPD",
            "role": "auxiliary_pdf_corpus",
            "purpose": "Raw-material EPD/support PDFs used as retrieval and labeling support material.",
        },
        "data/Coal_EPD": {
            "path": data_dir / "Coal_EPD",
            "role": "auxiliary_pdf_corpus",
            "purpose": "Coal quality/specification PDFs used as fuel reference support material.",
        },
    }

    summary: dict[str, dict[str, str]] = {}
    for label, info in audit_targets.items():
        directory = info["path"]
        subdirs = [
            child.name
            for child in sorted(directory.iterdir())
            if child.is_dir()
        ]
        summary[label] = {
            "role": info["role"],
            "purpose": info["purpose"],
            "file_types": _suffix_counts(directory),
            "subdirs": ", ".join(subdirs) if subdirs else "(none)",
        }
    return summary


def _format_audit_section(audit: dict[str, dict[str, str]]) -> str:
    lines = [
        "# Data Layout",
        "",
        "## Audit Summary",
        "",
        "- Main-flow data remains the product text -> NAICS -> EPA/USEEIO path built from `data/raw`, `data/processed`, and `data/splits`.",
        "- `data/Glass_EPD`, `data/Material_EPD`, and `data/Coal_EPD` are auxiliary PDF corpora. They support retrieval, reranking, and manual labeling, but they do not replace `data/raw/naics` or `data/raw/epa_factors`.",
        "- The main pre-existing organization gap was that the auxiliary PDFs had no stable document IDs, no shared metadata index, inconsistent subdirectory structure, and no document-level export for downstream retrieval/regression workflows.",
        "",
        "## Directory Roles",
        "",
        "| Directory | Role | Current files | Purpose |",
        "| --- | --- | --- | --- |",
    ]
    for label, info in audit.items():
        lines.append(
            f"| `{label}` | `{info['role']}` | {info['file_types']} | {info['purpose']} |"
        )

    lines.extend(
        [
            "",
            "## Directory Notes",
            "",
            f"- `data/raw`: {audit['data/raw']['subdirs']}",
            f"- `data/interim`: {audit['data/interim']['subdirs']}",
            f"- `data/processed`: {audit['data/processed']['subdirs']}",
            f"- `data/splits`: {audit['data/splits']['subdirs']}",
            f"- `data/Glass_EPD`: {audit['data/Glass_EPD']['subdirs']}",
            f"- `data/Material_EPD`: {audit['data/Material_EPD']['subdirs']}",
            f"- `data/Coal_EPD`: {audit['data/Coal_EPD']['subdirs']}",
            "",
            "## Main Flow vs Auxiliary Corpus",
            "",
            "- Main flow: product text samples from `data/raw/amazon_caml`, NAICS corpora from `data/raw/naics`, and EPA/USEEIO factors from `data/raw/epa_factors`.",
            "- Auxiliary corpus: PDF-derived support documents stored under `Glass_EPD`, `Material_EPD`, and `Coal_EPD` plus the generated `data/interim/aux_documents.csv` and `data/interim/aux_samples_for_labeling.csv`.",
            "- The auxiliary corpus is intended for reuse in retrieval, rerank, and regression experiments as side information or candidate-support material, not as a replacement label source.",
            "",
            "## How To Run",
            "",
            "Run the full auxiliary corpus preparation from the repository root:",
            "",
            "```bash",
            "python3 scripts/prepare_aux_corpus.py",
            "```",
            "",
            "Run the steps individually when needed:",
            "",
            "```bash",
            "python3 scripts/build_pdf_indexes.py",
            "python3 scripts/extract_aux_documents.py",
            "python3 scripts/build_aux_labeling_template.py",
            "```",
            "",
            "Outputs generated by the auxiliary workflow:",
            "",
            "- `data/Glass_EPD/index.csv`",
            "- `data/Material_EPD/index.csv`",
            "- `data/Coal_EPD/index.csv`",
            "- `data/interim/aux_documents.csv`",
            "- `data/interim/aux_samples_for_labeling.csv`",
            "- `data/interim/aux_text/*.txt`",
            "",
            "## OCR Hook",
            "",
            "- Default behavior only uses non-OCR parsing. If text extraction fails, the document is marked `needs_ocr` in `aux_documents.csv`.",
            "- To add OCR later, implement `open_match_lca.data.ocr_adapter.extract_text_with_ocr` and rerun:",
            "",
            "```bash",
            "python3 scripts/extract_aux_documents.py --enable-ocr",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_data_readme(repo_root: Path) -> Path:
    audit = audit_data_directories(repo_root)
    readme_path = _repo_data_dir(repo_root) / "README.md"
    readme_path.write_text(_format_audit_section(audit), encoding="utf-8")
    return readme_path


def infer_doc_type(file_name: str, relative_path: str) -> str:
    path_parts = Path(relative_path).parts[-2:]
    probe = " ".join(path_parts).replace("_", " ").replace("-", " ").lower()
    for token, label in DOC_TYPE_PATTERNS:
        if token in probe:
            return label
    return "other"


def infer_manufacturer(file_name: str) -> str:
    lower_name = file_name.lower()
    for token, label in GLASS_MANUFACTURERS.items():
        if token in lower_name:
            return label
    return ""


def infer_material_or_product(spec: CorpusSpec, relative_path: str, file_name: str) -> str:
    lower_path = relative_path.lower()
    lower_name = file_name.lower()
    normalized_name = lower_name.replace("_", " ").replace("-", " ")

    if spec.name == "glass":
        for token, label in GLASS_PRODUCT_PATTERNS:
            if token in normalized_name:
                return label
        return "glass"

    if spec.name == "material":
        for material in MATERIAL_SUBDIRS:
            if f"/{material}/" in lower_path.replace("\\", "/"):
                return material
        return "unknown"

    return "coal"


def _read_existing_index(index_path: Path) -> pd.DataFrame:
    if not index_path.exists():
        return pd.DataFrame(columns=INDEX_COLUMNS)
    frame = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    for column in INDEX_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[INDEX_COLUMNS].fillna("")


def _existing_doc_id_maps(existing: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    by_relative_path = dict(zip(existing["relative_path"], existing["doc_id"]))
    file_name_counts = existing["file_name"].value_counts().to_dict()
    by_file_name = {
        file_name: doc_id
        for file_name, doc_id in zip(existing["file_name"], existing["doc_id"])
        if file_name_counts.get(file_name, 0) == 1
    }
    return by_relative_path, by_file_name


def _next_doc_counter(existing: pd.DataFrame, prefix: str) -> int:
    max_value = 0
    for doc_id in existing["doc_id"]:
        if not doc_id.startswith(f"{prefix}_"):
            continue
        try:
            max_value = max(max_value, int(doc_id.split("_", maxsplit=1)[1]))
        except (IndexError, ValueError):
            continue
    return max_value + 1


def build_pdf_index(repo_root: Path, spec: CorpusSpec) -> Path:
    corpus_dir = _repo_data_dir(repo_root) / spec.root_dir
    index_path = corpus_dir / spec.index_name
    existing = _read_existing_index(index_path)
    existing_by_path, existing_by_file_name = _existing_doc_id_maps(existing)
    next_counter = _next_doc_counter(existing, spec.doc_prefix)
    assigned_doc_ids: set[str] = set(existing["doc_id"])

    records: list[dict[str, str]] = []
    for pdf_path in sorted(corpus_dir.rglob("*.pdf")):
        relative_path = str(pdf_path.relative_to(repo_root))
        file_name = pdf_path.name
        doc_id = existing_by_path.get(relative_path, "")
        if not doc_id:
            file_name_doc_id = existing_by_file_name.get(file_name, "")
            if file_name_doc_id and file_name_doc_id not in assigned_doc_ids:
                doc_id = file_name_doc_id
        if not doc_id:
            doc_id = f"{spec.doc_prefix}_{next_counter:04d}"
            next_counter += 1
        assigned_doc_ids.add(doc_id)

        notes = ""
        if "duplicate top-level" in file_name.lower():
            notes = "Top-level duplicate preserved during directory normalization."
        records.append(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "relative_path": relative_path,
                "doc_type": infer_doc_type(file_name, relative_path),
                "category_level_1": spec.category_level_1,
                "material_or_product": infer_material_or_product(spec, relative_path, file_name),
                "manufacturer": infer_manufacturer(file_name),
                "source_url": "",
                "notes": notes,
            }
        )

    frame = pd.DataFrame(records, columns=INDEX_COLUMNS).sort_values("doc_id").reset_index(drop=True)
    frame.to_csv(index_path, index=False, encoding="utf-8")
    return index_path


def build_all_pdf_indexes(repo_root: Path) -> list[Path]:
    return [build_pdf_index(repo_root, spec) for spec in AUX_CORPORA]


def load_all_pdf_indexes(repo_root: Path) -> pd.DataFrame:
    frames = []
    for spec in AUX_CORPORA:
        index_path = _repo_data_dir(repo_root) / spec.root_dir / spec.index_name
        frame = _read_existing_index(index_path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _write_raw_text(raw_text_dir: Path, doc_id: str, text: str) -> str:
    ensure_directory(raw_text_dir)
    raw_text_path = raw_text_dir / f"{doc_id}.txt"
    raw_text_path.write_text(text, encoding="utf-8")
    return str(raw_text_path)


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    return str(path.relative_to(repo_root))


def extract_aux_documents(
    repo_root: Path,
    *,
    enable_ocr: bool = False,
) -> Path:
    indexes = load_all_pdf_indexes(repo_root)
    interim_dir = _repo_data_dir(repo_root) / "interim"
    raw_text_dir = interim_dir / "aux_text"
    ensure_directory(interim_dir)
    ensure_directory(raw_text_dir)
    for stale_file in raw_text_dir.glob("*.txt"):
        stale_file.unlink()

    records: list[dict[str, str]] = []
    for row in indexes.to_dict(orient="records"):
        pdf_path = repo_root / row["relative_path"]
        parse_result = extract_pdf_text(pdf_path)
        text = parse_result.text
        parse_status = parse_result.parse_status

        if not text and enable_ocr:
            try:
                text = extract_text_with_ocr(pdf_path)
                if text:
                    parse_status = "parsed_ocr"
            except NotImplementedError:
                parse_status = "needs_ocr"

        title = parse_result.title or clean_filename_title(pdf_path)
        if not row["manufacturer"]:
            row["manufacturer"] = infer_manufacturer(row["file_name"])

        description_lines = candidate_description_lines(text)
        if parse_result.method != "pypdf" and not description_lines:
            text = ""
            parse_status = "needs_ocr"

        declared_unit = extract_declared_unit(text)
        thickness_mm_ref = extract_thickness_mm(text, title)
        mass_per_m2 = extract_mass_per_m2(text)
        system_boundary = extract_system_boundary(text)
        geography = extract_geography(text)
        year = extract_year(text, title)
        description_lines = candidate_description_lines(text)
        product_text = build_product_text(
            title=title,
            description_lines=description_lines,
            category_level_1=row["category_level_1"],
            material_or_product=row["material_or_product"],
            manufacturer=row["manufacturer"],
            declared_unit=declared_unit,
            thickness_mm_ref=thickness_mm_ref,
            mass_per_m2=mass_per_m2,
            system_boundary=system_boundary,
            geography=geography,
            year=year,
        )

        raw_text_path = ""
        if text:
            raw_text_path = _relative_to_repo(repo_root, Path(_write_raw_text(raw_text_dir, row["doc_id"], text)))

        records.append(
            {
                "doc_id": row["doc_id"],
                "title": title,
                "product_text": product_text,
                "category_level_1": row["category_level_1"],
                "material_or_product": row["material_or_product"],
                "manufacturer": row["manufacturer"],
                "source_type": row["doc_type"],
                "source_file": row["relative_path"],
                "declared_unit": declared_unit,
                "thickness_mm_ref": thickness_mm_ref,
                "mass_per_m2": mass_per_m2,
                "system_boundary": system_boundary,
                "geography": geography,
                "year": year,
                "raw_text_path": raw_text_path,
                "parse_status": parse_status,
            }
        )

    frame = pd.DataFrame(records, columns=AUX_DOCUMENT_COLUMNS).sort_values("doc_id").reset_index(drop=True)
    output_path = interim_dir / "aux_documents.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def _load_existing_label_template(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LABEL_TEMPLATE_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in LABEL_TEMPLATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[LABEL_TEMPLATE_COLUMNS].fillna("")


def _next_sample_counter(existing: pd.DataFrame) -> int:
    max_value = 0
    for sample_id in existing["sample_id"]:
        if not sample_id.startswith("aux_"):
            continue
        try:
            max_value = max(max_value, int(sample_id.split("_", maxsplit=1)[1]))
        except (IndexError, ValueError):
            continue
    return max_value + 1


def propose_naics_code(material_or_product: str, product_text: str) -> str:
    material = material_or_product.lower()
    product_text_lower = product_text.lower()

    if material == "silica_sand":
        return "212322"
    if material == "limestone":
        return "212312"
    if material == "coal":
        return "21211"
    if material in {
        "float_glass",
        "patterned_glass",
        "solar_glass",
        "thermally_toughened_float_glass",
        "low_iron_float_glass",
        "clearlite_glass",
    }:
        return "327211"
    if material == "glass" and any(token in product_text_lower for token in ("float glass", "patterned glass", "toughened glass")):
        return "327211"
    return ""


def build_labeling_template(repo_root: Path) -> Path:
    aux_documents_path = _repo_data_dir(repo_root) / "interim" / "aux_documents.csv"
    documents = pd.read_csv(aux_documents_path, dtype=str, keep_default_na=False).fillna("")
    output_path = _repo_data_dir(repo_root) / "interim" / "aux_samples_for_labeling.csv"
    existing = _load_existing_label_template(output_path)
    existing_by_doc_id = {row["doc_id"]: row for row in existing.to_dict(orient="records")}
    next_counter = _next_sample_counter(existing)

    rows: list[dict[str, str]] = []
    for record in documents.to_dict(orient="records"):
        existing_row = existing_by_doc_id.get(record["doc_id"], {})
        sample_id = existing_row.get("sample_id", "")
        if not sample_id:
            sample_id = f"aux_{next_counter:04d}"
            next_counter += 1
        rows.append(
            {
                "sample_id": sample_id,
                "doc_id": record["doc_id"],
                "product_text": record["product_text"],
                "category_level_1": record["category_level_1"],
                "material_or_product": record["material_or_product"],
                "proposed_naics_code": existing_row.get(
                    "proposed_naics_code",
                    propose_naics_code(record["material_or_product"], record["product_text"]),
                )
                or propose_naics_code(record["material_or_product"], record["product_text"]),
                "gold_naics_code": existing_row.get("gold_naics_code", ""),
                "label_status": existing_row.get("label_status", "unlabeled") or "unlabeled",
                "comments": existing_row.get("comments", ""),
            }
        )

    frame = pd.DataFrame(rows, columns=LABEL_TEMPLATE_COLUMNS).sort_values("sample_id").reset_index(drop=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return output_path
