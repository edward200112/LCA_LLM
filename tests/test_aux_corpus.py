from __future__ import annotations

from pathlib import Path

import pandas as pd

from open_match_lca.data.aux_corpus import (
    build_all_pdf_indexes,
    build_labeling_template,
    extract_aux_documents,
    normalize_aux_directories,
    write_data_readme,
)


def _make_repo_layout(tmp_path: Path) -> Path:
    repo_root = tmp_path
    for relative_dir in [
        "data/Glass_EPD",
        "data/Material_EPD",
        "data/Coal_EPD",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/splits",
    ]:
        (repo_root / relative_dir).mkdir(parents=True, exist_ok=True)
    return repo_root


def test_build_pdf_indexes_keeps_doc_ids_stable(tmp_path: Path) -> None:
    repo_root = _make_repo_layout(tmp_path)
    (repo_root / "data" / "Glass_EPD" / "Pilkington_Float_Glass.pdf").write_text(
        "Title: Pilkington Float Glass\nDeclared unit: 1 m2\nFloat glass for building products.",
        encoding="utf-8",
    )
    (repo_root / "data" / "Coal_EPD" / "QGESSDetailedCoalSpecifications_010112.pdf").write_text(
        "Quality Guidelines for Energy System Studies: Detailed Coal Specifications",
        encoding="utf-8",
    )

    normalize_aux_directories(repo_root)
    build_all_pdf_indexes(repo_root)
    first_index = pd.read_csv(repo_root / "data" / "Glass_EPD" / "index.csv", dtype=str, keep_default_na=False)

    build_all_pdf_indexes(repo_root)
    second_index = pd.read_csv(repo_root / "data" / "Glass_EPD" / "index.csv", dtype=str, keep_default_na=False)

    assert first_index["doc_id"].tolist() == second_index["doc_id"].tolist()
    assert (repo_root / "data" / "Coal_EPD" / "coal_specs" / "QGESSDetailedCoalSpecifications_010112.pdf").exists()


def test_extract_aux_documents_and_label_template(tmp_path: Path) -> None:
    repo_root = _make_repo_layout(tmp_path)
    (repo_root / "data" / "Glass_EPD" / "AGC 4 mm Low-Carbon Planibel Clearlite.pdf").write_text(
        "\n".join(
            [
                "AGC 4 mm Low-Carbon Planibel Clearlite",
                "Declared unit: 1 m2 of float glass",
                "Suitable for facade and window applications.",
                "Reference thickness 4 mm.",
                "System boundary: cradle-to-gate.",
                "Geography: Europe.",
                "Publication year 2024.",
                "Mass per area 10 kg/m2.",
            ]
        ),
        encoding="utf-8",
    )

    normalize_aux_directories(repo_root)
    build_all_pdf_indexes(repo_root)
    extract_aux_documents(repo_root)
    build_labeling_template(repo_root)
    write_data_readme(repo_root)

    aux_documents = pd.read_csv(repo_root / "data" / "interim" / "aux_documents.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(repo_root / "data" / "interim" / "aux_samples_for_labeling.csv", dtype=str, keep_default_na=False)

    assert aux_documents.loc[0, "thickness_mm_ref"] == "4"
    assert aux_documents.loc[0, "declared_unit"] == "1 m2 of float glass"
    assert aux_documents.loc[0, "parse_status"] in {"parsed_fallback", "parsed_pdf"}
    assert labels.loc[0, "proposed_naics_code"] == "327211"
    assert (repo_root / "data" / "README.md").exists()
