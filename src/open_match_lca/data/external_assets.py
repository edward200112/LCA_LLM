from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from open_match_lca.constants import PROJECT_ROOT
from open_match_lca.io_utils import ensure_directory


@dataclass(frozen=True)
class ExternalAsset:
    asset_id: str
    source_name: str
    url: str
    external_relpath: str
    raw_equivalent_relpath: str
    validation_kind: str
    required_for_case_study: bool = True
    manual_only: bool = False


EXTERNAL_ASSETS = [
    ExternalAsset(
        asset_id="amazon_40k_products",
        source_name="amazon_caml",
        url="https://raw.githubusercontent.com/amazon-science/carbon-assessment-with-ml/main/caml/data/40k_products_annotations.pkl",
        external_relpath="data/external_downloads/amazon_caml/40k_products_annotations.pkl",
        raw_equivalent_relpath="data/raw/amazon_caml/40k_products_annotations.pkl",
        validation_kind="binary",
    ),
    ExternalAsset(
        asset_id="amazon_6k_grocery_products",
        source_name="amazon_caml",
        url="https://raw.githubusercontent.com/amazon-science/carbon-assessment-with-ml/main/caml/data/6k_grocery_products_annotations.pkl",
        external_relpath="data/external_downloads/amazon_caml/6k_grocery_products_annotations.pkl",
        raw_equivalent_relpath="data/raw/amazon_caml/6k_grocery_products_annotations.pkl",
        validation_kind="binary",
    ),
    ExternalAsset(
        asset_id="amazon_naics_codes",
        source_name="amazon_caml",
        url="https://raw.githubusercontent.com/amazon-science/carbon-assessment-with-ml/main/caml/data/naics_codes.pkl",
        external_relpath="data/external_downloads/amazon_caml/naics_codes.pkl",
        raw_equivalent_relpath="data/raw/naics/naics_codes.pkl",
        validation_kind="binary",
    ),
    ExternalAsset(
        asset_id="naics_2017_structure",
        source_name="naics",
        url="https://www.census.gov/naics/2017NAICS/2017_NAICS_Structure.xlsx",
        external_relpath="data/external_downloads/naics/2017_NAICS_Structure.xlsx",
        raw_equivalent_relpath="data/external_downloads/naics/2017_NAICS_Structure.xlsx",
        validation_kind="xlsx",
    ),
    ExternalAsset(
        asset_id="naics_2022_structure",
        source_name="naics",
        url="https://www.census.gov/naics/2022NAICS/2022_NAICS_Structure.xlsx",
        external_relpath="data/external_downloads/naics/2022_NAICS_Structure.xlsx",
        raw_equivalent_relpath="data/external_downloads/naics/2022_NAICS_Structure.xlsx",
        validation_kind="xlsx",
        required_for_case_study=False,
    ),
    ExternalAsset(
        asset_id="naics_2022_to_2017_concordance",
        source_name="naics",
        url="https://www.census.gov/naics/concordances/2022_to_2017_NAICS.xlsx",
        external_relpath="data/external_downloads/naics/2022_to_2017_NAICS.xlsx",
        raw_equivalent_relpath="data/external_downloads/naics/2022_to_2017_NAICS.xlsx",
        validation_kind="xlsx",
        required_for_case_study=False,
    ),
    ExternalAsset(
        asset_id="naics_2017_to_2022_changes",
        source_name="naics",
        url="https://www.census.gov/naics/concordances/2017_to_2022_NAICS_Changes_Only.xlsx",
        external_relpath="data/external_downloads/naics/2017_to_2022_NAICS_Changes_Only.xlsx",
        raw_equivalent_relpath="data/external_downloads/naics/2017_to_2022_NAICS_Changes_Only.xlsx",
        validation_kind="xlsx",
        required_for_case_study=False,
    ),
    ExternalAsset(
        asset_id="epa_supply_chain_v13",
        source_name="epa",
        url="https://pasteur.epa.gov/uploads/10.23719/1531143/SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv",
        external_relpath="data/external_downloads/epa/SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv",
        raw_equivalent_relpath="data/raw/epa_factors/epa_naics_v13.csv",
        validation_kind="csv",
    ),
    ExternalAsset(
        asset_id="uslci_release_info",
        source_name="uslci",
        url="",
        external_relpath="data/external_downloads/uslci/release-downloads.md",
        raw_equivalent_relpath="data/raw/uslci",
        validation_kind="manual_only",
        required_for_case_study=False,
        manual_only=True,
    ),
]


DOWNLOAD_LOG_COLUMNS = [
    "asset_id",
    "source_name",
    "status",
    "url",
    "external_path",
    "raw_equivalent_path",
    "validated",
    "size_bytes",
    "checked_at",
    "error",
]


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or PROJECT_ROOT).resolve()


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _validate_download(path: Path, validation_kind: str) -> bool:
    if validation_kind == "manual_only":
        return False
    if validation_kind == "binary":
        return _is_nonempty_file(path)
    if validation_kind == "csv":
        if not _is_nonempty_file(path):
            return False
        pd.read_csv(path, nrows=5)
        return True
    if validation_kind == "xlsx":
        if not _is_nonempty_file(path):
            return False
        pd.read_excel(path, nrows=5)
        return True
    return _is_nonempty_file(path)


def _download_to_path(url: str, destination: Path) -> None:
    ensure_directory(destination.parent)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    tmp_path.write_bytes(payload)
    tmp_path.replace(destination)


def _row(asset: ExternalAsset, status: str, validated: bool, size_bytes: int, error: str = "") -> dict[str, object]:
    return {
        "asset_id": asset.asset_id,
        "source_name": asset.source_name,
        "status": status,
        "url": asset.url,
        "external_path": asset.external_relpath,
        "raw_equivalent_path": asset.raw_equivalent_relpath,
        "validated": validated,
        "size_bytes": size_bytes,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error": error,
    }


def fetch_external_assets(
    repo_root: Path | None = None,
    *,
    force: bool = False,
    include_optional: bool = True,
) -> pd.DataFrame:
    root = _repo_root(repo_root)
    rows: list[dict[str, object]] = []
    for asset in EXTERNAL_ASSETS:
        if not include_optional and not asset.required_for_case_study:
            rows.append(_row(asset, "skipped_optional", False, 0))
            continue

        external_path = root / asset.external_relpath
        raw_equivalent_path = root / asset.raw_equivalent_relpath
        raw_ready = _is_nonempty_file(raw_equivalent_path) or _is_nonempty_dir(raw_equivalent_path)
        if raw_ready and not force:
            size_bytes = raw_equivalent_path.stat().st_size if raw_equivalent_path.is_file() else 0
            rows.append(_row(asset, "skipped_existing_raw", True, size_bytes))
            continue

        if asset.manual_only:
            rows.append(_row(asset, "manual_source_required", False, 0, "No direct public artifact URL configured."))
            continue

        if _is_nonempty_file(external_path) and not force:
            try:
                validated = _validate_download(external_path, asset.validation_kind)
                rows.append(_row(asset, "skipped_existing_download", validated, external_path.stat().st_size))
            except Exception as exc:  # pragma: no cover - validation failures are data dependent
                rows.append(_row(asset, "failed_validation", False, external_path.stat().st_size, str(exc)))
            continue

        try:
            _download_to_path(asset.url, external_path)
            validated = _validate_download(external_path, asset.validation_kind)
            rows.append(_row(asset, "downloaded", validated, external_path.stat().st_size))
        except (URLError, OSError, ValueError, pd.errors.ParserError) as exc:
            rows.append(_row(asset, "download_failed", False, 0, str(exc)))

    frame = pd.DataFrame(rows, columns=DOWNLOAD_LOG_COLUMNS)
    log_path = root / "reports" / "audit" / "download_log.csv"
    ensure_directory(log_path.parent)
    frame.to_csv(log_path, index=False, encoding="utf-8")
    return frame
