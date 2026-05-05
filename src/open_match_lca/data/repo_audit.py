from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from open_match_lca.constants import PROJECT_ROOT
from open_match_lca.data.build_naics_corpus import NAICS_INPUT_REQUIRED_COLUMNS
from open_match_lca.data.parse_amazon_caml import AMAZON_CAML_ALT_COLUMNS, AMAZON_INPUT_REQUIRED_COLUMNS
from open_match_lca.data.parse_epa_factors import EPA_INPUT_REQUIRED_COLUMNS
from open_match_lca.io_utils import ensure_directory, load_yaml


AUDIT_ROOTS = [
    "data/raw",
    "data/processed",
    "data/interim",
    "data/splits",
    "reports",
    "models",
]

KEY_DATA_PATHS = [
    "data/raw/amazon_caml",
    "data/raw/epa_factors",
    "data/raw/naics",
    "data/raw/uslci",
    "data/raw/Glass_EPD",
    "data/raw/Material_EPD",
    "data/raw/glass_baseline",
    "data/interim/glass_factor_registry.csv",
    "data/interim/pv_glass_process_corpus.csv",
    "data/processed/products.parquet",
    "data/processed/epa_factors.parquet",
    "data/processed/naics_corpus.parquet",
    "data/processed/products_with_targets.parquet",
    "data/processed/uslci_processes.parquet",
]

MODEL_CANDIDATES = [
    "models/retriever/multilingual-e5-base",
    "models/reranker/ms-marco-MiniLM-L6-v2",
    "models/retriever/all-MiniLM-L6-v2",
    "models/reranker/bge-reranker-v2-m3",
]

TYPE_LABELS = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".md": "md",
    ".parquet": "parquet",
    ".pdf": "pdf",
    ".pkl": "pkl",
    ".py": "python",
    ".safetensors": "model",
    ".xlsx": "xlsx",
    "dir": "dir",
}


@dataclass(frozen=True)
class PathStatus:
    relative_path: str
    exists: bool
    item_type: str
    size_bytes: int
    modified_at: str
    likely_reusable: bool
    reuse_reason: str
    actual_alternative: str


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or PROJECT_ROOT).resolve()


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _human_file_type(path: Path) -> str:
    if path.is_dir():
        return TYPE_LABELS["dir"]
    return TYPE_LABELS.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "other")


def _likely_reusable(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.is_dir():
        has_children = any(path.iterdir())
        return has_children, "directory has contents" if has_children else "directory is empty"
    if path.stat().st_size <= 0:
        return False, "file is empty"
    if "reports/logs" in str(path).replace("\\", "/"):
        return False, "run log artifact"
    return True, "non-empty artifact"


def _actual_alternative(repo_root: Path, relative_path: str) -> str:
    candidate: Path | None = None
    if relative_path == "data/raw/Glass_EPD":
        candidate = repo_root / "data/Glass_EPD"
    elif relative_path == "data/raw/Material_EPD":
        candidate = repo_root / "data/Material_EPD"
    return str(candidate.relative_to(repo_root)) if candidate is not None and candidate.exists() else ""


def _path_status(repo_root: Path, relative_path: str) -> PathStatus:
    path = repo_root / relative_path
    exists = path.exists()
    likely_reusable, reuse_reason = _likely_reusable(path)
    return PathStatus(
        relative_path=relative_path,
        exists=exists,
        item_type=_human_file_type(path) if exists else "missing",
        size_bytes=path.stat().st_size if exists and path.is_file() else 0,
        modified_at=_iso_mtime(path) if exists else "",
        likely_reusable=likely_reusable,
        reuse_reason=reuse_reason,
        actual_alternative=_actual_alternative(repo_root, relative_path),
    )


def _collect_directory_inventory(repo_root: Path, relative_root: str) -> list[dict[str, object]]:
    root = repo_root / relative_root
    rows: list[dict[str, object]] = []
    if not root.exists():
        rows.append(
            {
                "path": relative_root,
                "item_type": "missing",
                "size_bytes": 0,
                "modified_at": "",
                "likely_reusable": False,
                "reuse_reason": "root path is missing",
            }
        )
        return rows
    for path in [root, *sorted(root.rglob("*"))]:
        if path.name == ".gitkeep":
            continue
        likely_reusable, reuse_reason = _likely_reusable(path)
        size_bytes = path.stat().st_size if path.is_file() else 0
        rows.append(
            {
                "path": str(path.relative_to(repo_root)),
                "item_type": _human_file_type(path),
                "size_bytes": size_bytes,
                "modified_at": _iso_mtime(path),
                "likely_reusable": likely_reusable,
                "reuse_reason": reuse_reason,
            }
        )
    return rows


def build_data_inventory(repo_root: Path | None = None) -> pd.DataFrame:
    root = _repo_root(repo_root)
    rows: list[dict[str, object]] = []
    for relative_root in AUDIT_ROOTS:
        rows.extend(_collect_directory_inventory(root, relative_root))
    frame = pd.DataFrame(rows)
    return frame.sort_values(["path", "item_type"]).reset_index(drop=True)


def _suffix_summary(path: Path) -> str:
    if not path.exists():
        return "missing"
    counts: dict[str, int] = {}
    for child in path.rglob("*"):
        if child.name == ".gitkeep":
            continue
        label = _human_file_type(child)
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "empty"
    return ", ".join(f"{key} x{counts[key]}" for key in sorted(counts))


def _load_full_config(repo_root: Path) -> dict:
    return load_yaml(repo_root / "configs/exp/full.yaml")


def _format_schema(columns: list[str]) -> str:
    return ", ".join(f"`{column}`" for column in columns)


def audit_repo_state(repo_root: Path | None = None) -> dict[str, object]:
    root = _repo_root(repo_root)
    config = _load_full_config(root)
    explicit_paths = [_path_status(root, relative_path) for relative_path in KEY_DATA_PATHS]
    model_paths = [_path_status(root, relative_path) for relative_path in MODEL_CANDIDATES]

    processed_required = [
        "data/processed/products.parquet",
        "data/processed/epa_factors.parquet",
        "data/processed/naics_corpus.parquet",
        "data/processed/products_with_targets.parquet",
    ]
    directory_summaries = {
        relative_root: _suffix_summary(root / relative_root)
        for relative_root in AUDIT_ROOTS
    }

    configured_retriever = str(config.get("dense_encoder_name", config.get("model_name", "")))
    configured_reranker = str(config.get("rerank_base_model", ""))
    configured_uslci = str(config.get("uslci_path", ""))
    return {
        "audit_generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "product_input_schema": AMAZON_INPUT_REQUIRED_COLUMNS,
        "product_alt_input_schema": AMAZON_CAML_ALT_COLUMNS,
        "epa_input_schema": EPA_INPUT_REQUIRED_COLUMNS,
        "naics_input_schema": NAICS_INPUT_REQUIRED_COLUMNS,
        "processed_required": processed_required,
        "configured_retriever_path": configured_retriever,
        "configured_reranker_path": configured_reranker,
        "configured_uslci_path": configured_uslci,
        "process_extension_default": bool(config.get("whether_process_extension", False)),
        "explicit_paths": explicit_paths,
        "model_paths": model_paths,
        "directory_summaries": directory_summaries,
    }


def _render_path_status_table(rows: list[PathStatus]) -> list[str]:
    lines = [
        "| Path | Exists | Type | Reusable | Notes | Alternative |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{path}` | {exists} | `{item_type}` | {reusable} | {reason} | {alt} |".format(
                path=row.relative_path,
                exists="yes" if row.exists else "no",
                item_type=row.item_type,
                reusable="yes" if row.likely_reusable else "no",
                reason=row.reuse_reason,
                alt=f"`{row.actual_alternative}`" if row.actual_alternative else "",
            )
        )
    return lines


def render_repo_state_audit(audit: dict[str, object]) -> str:
    explicit_paths = list(audit["explicit_paths"])
    model_paths = list(audit["model_paths"])
    configured_retriever = str(audit["configured_retriever_path"])
    configured_reranker = str(audit["configured_reranker_path"])
    configured_uslci = str(audit["configured_uslci_path"])
    retriever_exists = (PROJECT_ROOT / configured_retriever).exists()
    reranker_exists = (PROJECT_ROOT / configured_reranker).exists()
    uslci_exists = bool(configured_uslci) and (PROJECT_ROOT / configured_uslci).exists()

    lines = [
        "# Repository State Audit",
        "",
        f"- Generated at: `{audit['audit_generated_at']}`",
        "- Main task remains `product text -> NAICS -> EPA/USEEIO factor prediction`.",
        "- Current codebase already contains the merged mainline plus the auxiliary PDF corpus workflow under `data/Glass_EPD`, `data/Material_EPD`, and `data/Coal_EPD`.",
        "",
        "## Code Audit",
        "",
        f"- Product input schema accepted by `parse_amazon_caml`: {_format_schema(list(audit['product_input_schema']))}.",
        f"- Alternate product schema also supported: {_format_schema(list(audit['product_alt_input_schema']))}.",
        f"- EPA factor input schema accepted by `parse_epa_factors`: {_format_schema(list(audit['epa_input_schema']))}.",
        f"- NAICS input schema accepted by `build_naics_corpus`: {_format_schema(list(audit['naics_input_schema']))}.",
        "- Processed files written by `scripts/01_prepare_main_data.py`: `products.parquet`, `epa_factors.parquet`, `naics_corpus.parquet`, `products_with_targets.parquet`.",
        "- `scripts/02_make_splits.py` expects a product parquet and writes split parquet files.",
        "- `scripts/13_run_full_pipeline.py` delegates to the orchestration layer and resolves processed assets and split paths from config.",
        "",
        "## Full Config Audit",
        "",
        f"- `configs/exp/full.yaml` retriever path: `{configured_retriever}` (`{'present' if retriever_exists else 'missing'}`).",
        f"- `configs/exp/full.yaml` reranker path: `{configured_reranker}` (`{'present' if reranker_exists else 'missing'}`).",
        f"- `configs/exp/full.yaml` process extension default: `{audit['process_extension_default']}`.",
        f"- `configs/exp/full.yaml` USLCI path: `{configured_uslci or '(unset)'}` (`{'present' if uslci_exists else 'missing'}`).",
        "- Important behavior: the current orchestration layer still resolves `uslci_path` eagerly if it is present in config, so `full.yaml` is not runnable end-to-end unless that file exists.",
        "",
        "## Data Directory Summary",
        "",
    ]
    for relative_root, summary in dict(audit["directory_summaries"]).items():
        lines.append(f"- `{relative_root}`: {summary}")

    lines.extend(
        [
            "",
            "## Key Data Paths",
            "",
            *_render_path_status_table(explicit_paths),
            "",
            "## Model Paths",
            "",
            *_render_path_status_table(model_paths),
            "",
            "## Immediate Decisions",
            "",
            "- Reuse existing Amazon CaML-style raw data under `data/raw/amazon_caml`; no re-download is needed unless files are missing or empty.",
            "- Reuse existing main processed data and split files; they already exist.",
            "- `data/raw/uslci` is empty and `data/processed/uslci_processes.parquet` is missing, so the default process-extension path is incomplete.",
            "- Auxiliary PDF corpora already exist, but they live under `data/Glass_EPD` and `data/Material_EPD` rather than under `data/raw/...`; downstream scripts should consume the actual existing locations instead of redownloading duplicates.",
            "- Official 2017/2022 NAICS structure assets and the EPA v1.3 source CSV still need to be fetched if we want standardized case-study copies sourced from public upstream tables.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_repo_state_reports(repo_root: Path | None = None) -> tuple[Path, Path]:
    root = _repo_root(repo_root)
    reports_dir = root / "reports" / "audit"
    ensure_directory(reports_dir)
    inventory = build_data_inventory(root)
    audit = audit_repo_state(root)
    inventory_path = reports_dir / "data_inventory.csv"
    report_path = reports_dir / "repo_state_audit.md"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8")
    report_path.write_text(render_repo_state_audit(audit), encoding="utf-8")
    return report_path, inventory_path
