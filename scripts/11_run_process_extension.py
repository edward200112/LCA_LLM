from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.eval.eval_process_extension import export_process_extension_outputs
from open_match_lca.io_utils import ensure_directory, load_yaml, require_exists, write_jsonl, write_parquet
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.retrieval.process_extension import (
    load_process_exchanges,
    load_process_items,
    load_uslci_processes,
    recommend_process_items_with_audit,
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
    parser.add_argument("--config", required=False)
    parser.add_argument("--process_exchanges_path", required=False)
    parser.add_argument("--process_items_path", required=False)
    parser.add_argument("--item_top_k", type=int, required=False)
    parser.add_argument("--output_dir", required=True)
    return parser


def _resolve_optional_sidecar_path(
    configured_path: str | None,
    corpus_path: Path,
    default_filename: str,
) -> Path | None:
    if configured_path:
        return require_exists(Path(configured_path))
    candidate = corpus_path.parent / default_filename
    if candidate.exists():
        return candidate
    return None


def _resolve_item_recommendation_config(config_path: str | None) -> dict:
    defaults = {
        "domain_profile": "",
        "enable_domain_rerank": False,
        "enable_domain_filter": False,
        "domain_filter_min_score": 0.0,
        "domain_keep_topn_per_bucket": 0,
        "item_top_k": 10,
    }
    if not config_path:
        return defaults
    config = load_yaml(config_path)
    item_config = config.get("item_recommendation", {})
    if item_config and not isinstance(item_config, dict):
        raise RuntimeError(f"Invalid item_recommendation block in config: {config_path}")
    resolved = defaults.copy()
    resolved.update({key: config.get(key) for key in defaults if key in config})
    resolved.update(item_config or {})
    return resolved


def main() -> None:
    args = build_parser().parse_args()
    item_config = _resolve_item_recommendation_config(args.config)
    logger, _ = setup_run_logger("11_run_process_extension", LOGS_DIR, config_path=args.config)
    products_path = require_exists(Path(args.products_path))
    process_corpus_path = Path(args.uslci_path)
    if not process_corpus_path.exists():
        raise FileNotFoundError(
            f"Process corpus data not found at {process_corpus_path}. "
            "This feature is an optional extension; the main experiment is unaffected."
        )
    products = pd.read_parquet(products_path)
    uslci_processes = load_uslci_processes(process_corpus_path)
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    split_name = products_path.stem
    retriever_name = Path(args.retriever_ckpt).name if args.retriever_ckpt != "bm25" else "bm25"
    item_top_k = int(args.item_top_k if args.item_top_k is not None else item_config.get("item_top_k", 10))

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

    resolved_process_exchanges_path = _resolve_optional_sidecar_path(
        args.process_exchanges_path,
        process_corpus_path,
        "process_exchanges.parquet",
    )
    resolved_process_items_path = _resolve_optional_sidecar_path(
        args.process_items_path,
        process_corpus_path,
        "process_items_standardized.parquet",
    )
    item_recommendation_output = None
    if resolved_process_exchanges_path is not None or resolved_process_items_path is not None:
        if resolved_process_exchanges_path is None or resolved_process_items_path is None:
            raise FileNotFoundError(
                "Process item recommendation requires both process_exchanges and process_items assets. "
                f"Resolved process_exchanges_path={resolved_process_exchanges_path}, "
                f"process_items_path={resolved_process_items_path}"
            )
        process_exchanges = load_process_exchanges(resolved_process_exchanges_path)
        process_items = load_process_items(resolved_process_items_path)
        item_recommendations, item_recommendation_audit = recommend_process_items_with_audit(
            final_records,
            process_exchanges,
            process_items,
            top_k=item_top_k,
            domain_config=item_config,
        )
        item_recommendation_output = output_dir / f"process_item_recommendations_{split_name}_{retriever_name}.parquet"
        write_parquet(item_recommendations, item_recommendation_output)
        logger.info(
            "process_item_recommendations_written",
            extra={"structured": {"output_path": str(item_recommendation_output)}},
        )
        domain_profile = str(item_config.get("domain_profile", ""))
        if domain_profile:
            audit_output = output_dir / f"process_item_recommendations_audit_{split_name}_{retriever_name}.parquet"
            write_parquet(item_recommendation_audit, audit_output)
            logger.info(
                "process_item_recommendations_audit_written",
                extra={"structured": {"output_path": str(audit_output), "domain_profile": domain_profile}},
            )
        else:
            audit_output = None
    else:
        audit_output = None

    log_final_metrics(
        logger,
        {
            "queries": len(final_records),
            "has_silver_labels": bool(has_silver_labels),
            "output_path": str(review_or_metrics_path),
            "item_recommendation_output": str(item_recommendation_output) if item_recommendation_output else None,
            "item_recommendation_audit_output": str(audit_output) if audit_output else None,
            "domain_profile": str(item_config.get("domain_profile", "")),
        },
    )


if __name__ == "__main__":
    main()
