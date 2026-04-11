from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.eval.eval_process_extension import export_process_extension_outputs
from open_match_lca.io_utils import ensure_directory, require_exists, write_jsonl
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.retrieval.process_extension import (
    load_uslci_processes,
    rerank_process_candidates,
    retrieve_process_candidates,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products_path", required=True)
    parser.add_argument("--uslci_path", required=True)
    parser.add_argument("--prefilter_by_naics", choices=["true", "false"], required=True)
    parser.add_argument("--retriever_ckpt", required=True)
    parser.add_argument("--reranker_ckpt", required=False)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("11_run_process_extension", LOGS_DIR)
    products_path = require_exists(Path(args.products_path))
    uslci_path = Path(args.uslci_path)
    if not uslci_path.exists():
        raise FileNotFoundError(
            f"USLCI data not found at {uslci_path}. "
            "This feature is an optional extension; the main experiment is unaffected."
        )
    products = pd.read_parquet(products_path)
    uslci_processes = load_uslci_processes(uslci_path)
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    split_name = products_path.stem
    retriever_name = Path(args.retriever_ckpt).name if args.retriever_ckpt != "bm25" else "bm25"

    records = retrieve_process_candidates(
        products_frame=products,
        uslci_frame=uslci_processes,
        retriever_ckpt=args.retriever_ckpt,
        top_k=10,
        prefilter_by_naics=args.prefilter_by_naics == "true",
    )
    base_output = output_dir / f"process_topk_{split_name}_{retriever_name}.jsonl"
    write_jsonl(records, base_output)
    logger.info("process_candidates_written", extra={"structured": {"output_path": str(base_output)}})

    final_records = records
    if args.reranker_ckpt:
        final_records = rerank_process_candidates(
            retrieval_records=records,
            model_name_or_path=args.reranker_ckpt,
            batch_size=8,
            top_k=10,
        )
        reranked_output = output_dir / f"process_topk_{split_name}_{retriever_name}_reranked.jsonl"
        write_jsonl(final_records, reranked_output)
        logger.info("process_reranked_written", extra={"structured": {"output_path": str(reranked_output)}})

    has_silver_labels = "gold_process_uuid" in products.columns and products["gold_process_uuid"].notna().any()
    review_or_metrics_path = (
        output_dir / "process_extension_metrics.json"
        if has_silver_labels
        else output_dir / "process_review_pack.parquet"
    )
    export_process_extension_outputs(final_records, has_silver_labels, str(review_or_metrics_path))
    log_final_metrics(
        logger,
        {
            "queries": len(final_records),
            "has_silver_labels": bool(has_silver_labels),
            "output_path": str(review_or_metrics_path),
        },
    )


if __name__ == "__main__":
    main()
