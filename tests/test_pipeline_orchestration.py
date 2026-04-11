from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from open_match_lca.pipeline.orchestration import apply_ablation, build_pipeline_manifest


def test_apply_ablation_overrides() -> None:
    base = {
        "retrieval_mode": "hybrid",
        "whether_rerank": True,
        "whether_regression": True,
        "whether_process_extension": True,
        "whether_uncertainty": True,
        "use_hierarchy_features": True,
    }
    updated = apply_ablation(base, "uncertainty_off")
    assert updated["whether_uncertainty"] is False
    updated = apply_ablation(base, "hierarchy_features_off")
    assert updated["use_hierarchy_features"] is False


def test_build_pipeline_manifest(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    processed_dir = tmp_path / "processed"
    split_dir.mkdir()
    processed_dir.mkdir()
    sample = pd.DataFrame([{"product_id": "p1", "text": "sample", "gold_naics_code": "111111"}])
    sample.to_parquet(split_dir / "random_stratified_train.parquet", index=False)
    sample.to_parquet(split_dir / "random_stratified_dev.parquet", index=False)
    sample.to_parquet(split_dir / "random_stratified_test.parquet", index=False)
    pd.DataFrame([{"naics_code": "111111", "naics_text": "sample class"}]).to_parquet(
        processed_dir / "naics_corpus.parquet", index=False
    )
    pd.DataFrame([{"naics_code": "111111", "factor_value": 1.0}]).to_parquet(
        processed_dir / "epa_factors.parquet", index=False
    )
    config = {
        "split_type": "random_stratified",
        "splits_dir": str(split_dir),
        "processed_dir": str(processed_dir),
        "retrieval_mode": "hybrid",
        "whether_rerank": True,
        "whether_regression": True,
        "whether_process_extension": False,
        "whether_uncertainty": True,
    }
    manifest = build_pipeline_manifest(config, 13, tmp_path / "runs")
    assert manifest["seed"] == 13
    assert any(step["name"] == "reranker" and step["enabled"] for step in manifest["steps"])


def test_ablation_cli_dry_run(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    split_dir = tmp_path / "splits"
    processed_dir = tmp_path / "processed"
    split_dir.mkdir()
    processed_dir.mkdir()
    sample = pd.DataFrame([{"product_id": "p1", "text": "sample", "gold_naics_code": "111111"}])
    sample.to_parquet(split_dir / "random_stratified_train.parquet", index=False)
    sample.to_parquet(split_dir / "random_stratified_dev.parquet", index=False)
    sample.to_parquet(split_dir / "random_stratified_test.parquet", index=False)
    pd.DataFrame([{"naics_code": "111111", "naics_text": "sample class"}]).to_parquet(
        processed_dir / "naics_corpus.parquet", index=False
    )
    pd.DataFrame([{"naics_code": "111111", "factor_value": 1.0}]).to_parquet(
        processed_dir / "epa_factors.parquet", index=False
    )
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "split_type": "random_stratified",
                "splits_dir": str(split_dir),
                "processed_dir": str(processed_dir),
                "corpus_path": str(processed_dir / "naics_corpus.parquet"),
                "epa_factors_path": str(processed_dir / "epa_factors.parquet"),
                "retrieval_mode": "hybrid",
                "whether_rerank": False,
                "whether_regression": False,
                "whether_process_extension": False,
                "whether_uncertainty": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "ablation"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "10_run_ablation.py"),
            "--exp_config",
            str(config_path),
            "--seed",
            "13",
            "--output_dir",
            str(output_dir),
            "--ablation",
            "bm25_only",
            "--dry_run",
        ],
        check=True,
    )
    manifest = json.loads((output_dir / "ablation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"][0]["ablation"] == "bm25_only"
    assert (output_dir / "bm25_only" / "pipeline_manifest.json").exists()
