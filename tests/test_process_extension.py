from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from open_match_lca.eval.eval_process_extension import evaluate_process_extension
from open_match_lca.retrieval.process_extension import retrieve_process_candidates


def test_process_extension_metrics_with_silver_labels() -> None:
    records = [
        {
            "product_id": "p1",
            "query_text": "metal chair",
            "gold_process_uuid": "proc-1",
            "candidates": [
                {"process_uuid": "proc-1", "score": 1.0},
                {"process_uuid": "proc-2", "score": 0.5},
            ],
        }
    ]
    metrics = evaluate_process_extension(records, has_silver_labels=True)
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr@10"] == 1.0


def test_process_extension_bm25_retrieval() -> None:
    products = pd.DataFrame(
        [
            {"product_id": "p1", "text": "steel wire", "gold_process_uuid": "proc-1"},
        ]
    )
    uslci = pd.DataFrame(
        [
            {
                "process_uuid": "proc-1",
                "process_name": "steel wire drawing",
                "category_path": "metals",
                "geography": "US",
                "reference_flow_name": "wire",
                "reference_flow_unit": "kg",
                "process_text": "steel wire drawing process",
                "source_release": "v1",
            },
            {
                "process_uuid": "proc-2",
                "process_name": "paper making",
                "category_path": "paper",
                "geography": "US",
                "reference_flow_name": "paper",
                "reference_flow_unit": "kg",
                "process_text": "paper making process",
                "source_release": "v1",
            },
        ]
    )
    records = retrieve_process_candidates(products, uslci, retriever_ckpt="bm25", top_k=2)
    assert records[0]["candidates"][0]["process_uuid"] == "proc-1"


def test_process_extension_cli_review_pack(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    products = pd.DataFrame(
        [
            {"product_id": "p1", "text": "steel wire"},
        ]
    )
    uslci = pd.DataFrame(
        [
            {
                "process_uuid": "proc-1",
                "process_name": "steel wire drawing",
                "category_path": "metals",
                "geography": "US",
                "reference_flow_name": "wire",
                "reference_flow_unit": "kg",
                "process_text": "steel wire drawing process",
                "source_release": "v1",
            }
        ]
    )
    products_path = tmp_path / "products.parquet"
    uslci_path = tmp_path / "uslci.parquet"
    output_dir = tmp_path / "outputs"
    products.to_parquet(products_path, index=False)
    uslci.to_parquet(uslci_path, index=False)
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "11_run_process_extension.py"),
            "--products_path",
            str(products_path),
            "--uslci_path",
            str(uslci_path),
            "--prefilter_by_naics",
            "false",
            "--retriever_ckpt",
            "bm25",
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )
    assert (output_dir / "process_review_pack.parquet").exists()
    assert (output_dir / "process_topk_products_bm25.jsonl").exists()
