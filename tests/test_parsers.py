from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from open_match_lca.data.build_naics_corpus import build_naics_corpus
from open_match_lca.data.parse_amazon_caml import parse_amazon_caml
from open_match_lca.data.parse_epa_factors import parse_epa_factors


def test_parse_amazon_caml_missing_columns(tmp_path: Path) -> None:
    raw_dir = tmp_path / "amazon"
    raw_dir.mkdir()
    pd.DataFrame({"product_id": ["1"], "title": ["chair"]}).to_csv(raw_dir / "products.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        parse_amazon_caml(str(raw_dir))


def test_parsers_roundtrip(tmp_path: Path) -> None:
    amazon_dir = tmp_path / "amazon"
    epa_dir = tmp_path / "epa"
    naics_dir = tmp_path / "naics"
    amazon_dir.mkdir()
    epa_dir.mkdir()
    naics_dir.mkdir()

    pd.DataFrame(
        [
            {"product_id": "p1", "title": "Steel chair 20cm", "description": "metal frame", "gold_naics_code": "337127"},
            {"product_id": "p2", "title": "Wood desk", "description": "oak desk", "gold_naics_code": "337211"},
        ]
    ).to_csv(amazon_dir / "products.csv", index=False)
    pd.DataFrame(
        [
            {
                "naics_code": "337127",
                "factor_value": 1.5,
                "factor_unit": "kg/$",
                "with_margins": 1.7,
                "without_margins": 1.5,
                "source_year": 2020,
                "useeio_code": "A1",
            }
        ]
    ).to_csv(epa_dir / "factors.csv", index=False)
    pd.DataFrame(
        [
            {"naics_code": "337127", "naics_title": "Institutional Furniture"},
            {"naics_code": "337211", "naics_title": "Wood Office Furniture"},
        ]
    ).to_csv(naics_dir / "naics.csv", index=False)

    products = parse_amazon_caml(str(amazon_dir))
    epa = parse_epa_factors(str(epa_dir))
    naics = build_naics_corpus(str(naics_dir))

    assert set(["text", "text_len", "has_numeric_tokens"]).issubset(products.columns)
    assert list(epa["naics_code"]) == ["337127"]
    assert naics["naics_code_2"].tolist() == ["33", "33"]


def test_smoke_pipeline_cli(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_root = tmp_path / "raw"
    amazon_dir = raw_root / "amazon_caml"
    epa_dir = raw_root / "epa_factors"
    naics_dir = raw_root / "naics"
    out_dir = tmp_path / "processed"
    split_dir = tmp_path / "splits"
    amazon_dir.mkdir(parents=True)
    epa_dir.mkdir(parents=True)
    naics_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"product_id": "p1", "title": "Steel chair 20cm", "description": "metal frame", "gold_naics_code": "337127"},
            {"product_id": "p2", "title": "Wood desk", "description": "oak desk", "gold_naics_code": "337211"},
            {"product_id": "p3", "title": "Office lamp", "description": "aluminum lamp", "gold_naics_code": "335139"},
            {"product_id": "p4", "title": "Office lamp", "description": "aluminum lamp", "gold_naics_code": "335139"},
            {"product_id": "p5", "title": "Plastic bin 10L", "description": "storage box", "gold_naics_code": "326199"},
            {"product_id": "p6", "title": "Paper notebook", "description": "ruled paper", "gold_naics_code": "322230"},
        ]
    ).to_csv(amazon_dir / "products.csv", index=False)
    pd.DataFrame(
        [
            {"naics_code": "337127", "factor_value": 1.5, "factor_unit": "kg/$", "with_margins": 1.7, "without_margins": 1.5, "source_year": 2020, "useeio_code": "A1"},
            {"naics_code": "337211", "factor_value": 2.1, "factor_unit": "kg/$", "with_margins": 2.3, "without_margins": 2.1, "source_year": 2020, "useeio_code": "A2"},
            {"naics_code": "335139", "factor_value": 3.1, "factor_unit": "kg/$", "with_margins": 3.2, "without_margins": 3.1, "source_year": 2020, "useeio_code": "A3"},
            {"naics_code": "326199", "factor_value": 0.9, "factor_unit": "kg/$", "with_margins": 1.0, "without_margins": 0.9, "source_year": 2020, "useeio_code": "A4"},
            {"naics_code": "322230", "factor_value": 0.7, "factor_unit": "kg/$", "with_margins": 0.8, "without_margins": 0.7, "source_year": 2020, "useeio_code": "A5"},
        ]
    ).to_csv(epa_dir / "factors.csv", index=False)
    pd.DataFrame(
        [
            {"naics_code": "337127", "naics_title": "Institutional Furniture"},
            {"naics_code": "337211", "naics_title": "Wood Office Furniture"},
            {"naics_code": "335139", "naics_title": "Electric Lamps"},
            {"naics_code": "326199", "naics_title": "Other Plastics Products"},
            {"naics_code": "322230", "naics_title": "Stationery Products"},
        ]
    ).to_csv(naics_dir / "naics.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "01_prepare_main_data.py"),
            "--amazon_dir",
            str(amazon_dir),
            "--epa_dir",
            str(epa_dir),
            "--naics_dir",
            str(naics_dir),
            "--out_dir",
            str(out_dir),
            "--config",
            str(project_root / "configs" / "data" / "main.yaml"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "02_make_splits.py"),
            "--input_products",
            str(out_dir / "products.parquet"),
            "--split_type",
            "cluster_ood",
            "--seed",
            "13",
            "--out_dir",
            str(split_dir),
        ],
        check=True,
    )

    assert (out_dir / "products.parquet").exists()
    assert (out_dir / "epa_factors.parquet").exists()
    assert (out_dir / "naics_corpus.parquet").exists()
    assert (split_dir / "cluster_ood_train.parquet").exists()
    assert (split_dir / "cluster_ood_dev.parquet").exists()
    assert (split_dir / "cluster_ood_test.parquet").exists()
