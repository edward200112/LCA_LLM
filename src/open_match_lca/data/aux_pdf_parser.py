from __future__ import annotations

import importlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_TITLE_HEX_PATTERN = re.compile(rb"/Title\s*<([0-9A-Fa-f]+)>", re.DOTALL)
_TITLE_TEXT_PATTERN = re.compile(rb"/Title\s*\((.*?)\)", re.DOTALL)
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_THICKNESS_PATTERN = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*mm\b")
_MASS_PER_AREA_PATTERNS = [
    re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*kg\s*/\s*m(?:2|²)\b"),
    re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*kg\s+per\s+m(?:2|²)\b"),
]
_DECLARED_UNIT_PATTERNS = [
    re.compile(r"(?i)declared unit\s*[:\-]\s*([^\n\r.;]{1,120})"),
    re.compile(r"(?i)functional unit\s*[:\-]\s*([^\n\r.;]{1,120})"),
]
_SYSTEM_BOUNDARY_PATTERNS = [
    re.compile(r"(?i)\bcradle[- ]to[- ]gate\b"),
    re.compile(r"(?i)\bcradle[- ]to[- ]grave\b"),
    re.compile(r"(?i)\bA1\s*-\s*A3\b"),
    re.compile(r"(?i)\bA1\s*to\s*A3\b"),
    re.compile(r"(?i)\bA1\s*-\s*A5\b"),
    re.compile(r"(?i)\bA1\s*-\s*C4\b"),
]
_GEOGRAPHY_KEYWORDS = [
    "Global",
    "Europe",
    "European Union",
    "North America",
    "United States",
    "United Kingdom",
    "Turkey",
    "China",
    "India",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Poland",
    "Sweden",
]
_BORING_TITLE_PATTERNS = [
    re.compile(r"(?i)^back v \d"),
    re.compile(r"(?i)^epd document[_\s-]*epd"),
]
_DROP_LINE_PREFIXES = (
    "%PDF",
    "endobj",
    "stream",
    "endstream",
    "obj",
    "xref",
    "trailer",
    "startxref",
    "%%EOF",
)


@dataclass(frozen=True)
class PdfParseResult:
    text: str
    title: str
    method: str
    parse_status: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_filename_title(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def _decode_pdf_title_bytes(raw_value: bytes) -> str:
    candidate = raw_value.strip()
    if not candidate:
        return ""
    if candidate.startswith(b"\xfe\xff") or candidate.startswith(b"\xff\xfe"):
        for encoding in ("utf-16-be", "utf-16-le", "utf-16"):
            try:
                return candidate.decode(encoding, errors="ignore").strip("\x00 ").strip()
            except UnicodeDecodeError:
                continue
    for encoding in ("utf-8", "latin-1"):
        try:
            return candidate.decode(encoding, errors="ignore").strip("\x00 ").strip()
        except UnicodeDecodeError:
            continue
    return ""


def extract_pdf_title(path: Path) -> str:
    raw = path.read_bytes()
    match = _TITLE_HEX_PATTERN.search(raw)
    if match:
        try:
            decoded = bytes.fromhex(match.group(1).decode("ascii", errors="ignore"))
            title = _decode_pdf_title_bytes(decoded)
            if title and not is_low_quality_title(title):
                return normalize_whitespace(title)
        except ValueError:
            pass

    match = _TITLE_TEXT_PATTERN.search(raw)
    if match:
        title = _decode_pdf_title_bytes(match.group(1))
        if title and not is_low_quality_title(title):
            return normalize_whitespace(title)

    return clean_filename_title(path)


def is_low_quality_title(title: str) -> bool:
    candidate = normalize_whitespace(title)
    if not candidate:
        return True
    return any(pattern.search(candidate) for pattern in _BORING_TITLE_PATTERNS)


def _load_pdf_reader() -> type | None:
    for module_name, attr_name in (("pypdf", "PdfReader"), ("PyPDF2", "PdfReader")):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        reader_type = getattr(module, attr_name, None)
        if reader_type is not None:
            return reader_type
    return None


def _extract_text_with_pdf_reader(path: Path) -> str:
    reader_type = _load_pdf_reader()
    if reader_type is None:
        return ""
    reader = reader_type(str(path))
    parts: list[str] = []
    for page in getattr(reader, "pages", []):
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _looks_meaningful_line(line: str) -> bool:
    if len(line) < 4:
        return False
    if line.startswith(_DROP_LINE_PREFIXES):
        return False
    lower_line = line.lower()
    if any(
        token in lower_line
        for token in (
            "xmpgimg:image",
            "rdf:description",
            "fontfile",
            "filter/flatedecode",
            "xmlns:",
            "x:xmpmeta",
            "rdf:rdf",
            "/type/",
            " obj",
            "endobj",
            "stream",
            "endstream",
        )
    ):
        return False
    if any(symbol in line for symbol in ("<<", ">>", "/Type", "/Pages", "/Catalog", "<x:", "</")):
        return False
    alpha_chars = sum(char.isalpha() for char in line)
    if alpha_chars < 3:
        return False
    punctuation = sum(char in "{}[]<>/@\\" for char in line)
    if punctuation > max(5, len(line) // 4):
        return False
    if line.count("/") > max(2, len(line) // 20):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9&().,\-]*", line)
    if len(words) < 2 and " " not in line:
        return False
    return True


def _extract_text_with_strings(path: Path) -> str:
    try:
        result = subprocess.run(
            ["strings", str(path)],
            check=True,
            capture_output=True,
        )
    except Exception:
        return ""

    raw_text = result.stdout.decode("utf-8", errors="ignore")
    kept_lines: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_text.splitlines():
        line = normalize_whitespace(raw_line)
        if not _looks_meaningful_line(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _is_meaningful_text(text: str) -> bool:
    words = re.findall(r"[A-Za-z]{3,}", text)
    return len(words) >= 20 and len(normalize_whitespace(text)) >= 120


def extract_pdf_text(path: str | Path) -> PdfParseResult:
    pdf_path = Path(path)
    title = extract_pdf_title(pdf_path)

    pdf_text = _extract_text_with_pdf_reader(pdf_path)
    if _is_meaningful_text(pdf_text):
        return PdfParseResult(
            text=pdf_text.strip(),
            title=title,
            method="pypdf",
            parse_status="parsed_pdf",
        )

    fallback_text = _extract_text_with_strings(pdf_path)
    if _is_meaningful_text(fallback_text):
        return PdfParseResult(
            text=fallback_text.strip(),
            title=title,
            method="strings",
            parse_status="parsed_fallback",
        )

    return PdfParseResult(
        text="",
        title=title,
        method="none",
        parse_status="needs_ocr",
    )


def candidate_description_lines(text: str, limit: int = 4) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    content_cues = (
        "declared",
        "unit",
        "product",
        "glass",
        "sand",
        "silica",
        "limestone",
        "dolomite",
        "feldspar",
        "coal",
        "ash",
        "float",
        "patterned",
        "toughened",
        "solar",
        "facade",
        "window",
        "application",
        "system boundary",
        "mass",
        "manufacturer",
        "description",
    )
    for raw_line in text.splitlines():
        line = normalize_whitespace(raw_line)
        if len(line) < 24 or len(line) > 220:
            continue
        lower_line = line.lower()
        if any(
            token in lower_line
            for token in (
                "rdf:",
                "xmp",
                "xmlns",
                "creatortool",
                "documentid",
                "instanceid",
                "originaldocumentid",
                "parseType".lower(),
                "pdf library",
                "microsoft.",
                "adobe ",
                "photoshop",
                "tiff:",
                "crs:",
                "rawfilename",
            )
        ):
            continue
        if any(symbol in line for symbol in ("<", ">", "{", "}", "[", "]")):
            continue
        if sum(char in "#%\\/" for char in line) > 2:
            continue
        if sum(char.isalpha() for char in line) < 10:
            continue
        word_tokens = re.findall(r"[A-Za-z]{3,}", line)
        if len(word_tokens) < 4:
            continue
        if not any(cue in lower_line for cue in content_cues):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def extract_declared_unit(text: str) -> str:
    for line in candidate_description_lines(text, limit=20):
        for pattern in _DECLARED_UNIT_PATTERNS:
            match = pattern.search(line)
            if match:
                return normalize_whitespace(match.group(1))
    return ""


def extract_thickness_mm(text: str, title: str = "") -> str:
    match = _THICKNESS_PATTERN.search(title)
    if match:
        value = match.group(1)
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        return value

    for line in candidate_description_lines(text, limit=20):
        lowered = line.lower()
        if "mm" not in lowered:
            continue
        if not any(token in lowered for token in ("thick", "thickness", "glass", "coated", "clear", "solar", "patterned")):
            continue
        match = _THICKNESS_PATTERN.search(line)
        if match:
            value = match.group(1)
            if "." in value:
                value = value.rstrip("0").rstrip(".")
            return value
    return ""


def extract_mass_per_m2(text: str) -> str:
    for line in candidate_description_lines(text, limit=20):
        for pattern in _MASS_PER_AREA_PATTERNS:
            match = pattern.search(line)
            if match:
                value = match.group(1)
                return value.rstrip("0").rstrip(".") if "." in value else value
    return ""


def extract_system_boundary(text: str) -> str:
    for line in candidate_description_lines(text, limit=20):
        for pattern in _SYSTEM_BOUNDARY_PATTERNS:
            match = pattern.search(line)
            if match:
                return normalize_whitespace(match.group(0))
    return ""


def extract_geography(text: str) -> str:
    for line in candidate_description_lines(text, limit=20):
        lowered = line.lower()
        for keyword in _GEOGRAPHY_KEYWORDS:
            if keyword.lower() in lowered:
                return keyword
    return ""


def extract_year(text: str, title: str = "") -> str:
    candidates = candidate_description_lines(text, limit=20)
    if title:
        candidates.insert(0, title)

    valid_years: list[str] = []
    for line in candidates:
        if len(line) > 200:
            continue
        if line != title and not any(token in line.lower() for token in ("year", "date", "published", "issue", "valid")):
            continue
        valid_years.extend(_YEAR_PATTERN.findall(line))

    return max(valid_years) if valid_years else ""


def build_product_text(
    *,
    title: str,
    description_lines: list[str],
    category_level_1: str,
    material_or_product: str,
    manufacturer: str,
    declared_unit: str,
    thickness_mm_ref: str,
    mass_per_m2: str,
    system_boundary: str,
    geography: str,
    year: str,
) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}.")
    if category_level_1 or material_or_product:
        category_text = f"Category: {category_level_1}"
        if material_or_product:
            category_text += f"; material or product: {material_or_product.replace('_', ' ')}"
        parts.append(category_text + ".")
    if manufacturer:
        parts.append(f"Manufacturer: {manufacturer}.")
    if description_lines:
        parts.append("Description: " + " ".join(description_lines) + ".")

    specs: list[str] = []
    if declared_unit:
        specs.append(f"declared unit {declared_unit}")
    if thickness_mm_ref:
        specs.append(f"reference thickness {thickness_mm_ref} mm")
    if mass_per_m2:
        specs.append(f"mass per area {mass_per_m2} kg/m2")
    if system_boundary:
        specs.append(f"system boundary {system_boundary}")
    if geography:
        specs.append(f"geography {geography}")
    if year:
        specs.append(f"year {year}")
    if specs:
        parts.append("Key specs: " + "; ".join(specs) + ".")

    return normalize_whitespace(" ".join(parts))
