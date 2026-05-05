from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.openlca_hybrid import retrieve_process_candidates_from_registry
from open_match_lca.io_utils import ensure_directory, load_yaml, write_jsonl
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", default="data/processed/pv_glass_cases.parquet")
    parser.add_argument("--process_corpus_path", default="data/processed/openlca_process_registry.parquet")
    parser.add_argument("--config", default="configs/exp/full_pv_glass_hybrid.yaml")
    parser.add_argument("--output_dir", default="reports/case_study/pv_glass/process_retrieval")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_yaml(repo_root / args.config if not Path(args.config).is_absolute() else Path(args.config))
    logger, _ = setup_run_logger("08b_retrieve_process_candidates", LOGS_DIR, config_path=args.config)

    cases_path = repo_root / args.cases_path if not Path(args.cases_path).is_absolute() else Path(args.cases_path)
    process_path = (
        repo_root / args.process_corpus_path if not Path(args.process_corpus_path).is_absolute() else Path(args.process_corpus_path)
    )
    output_dir = repo_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    ensure_directory(output_dir)

    cases_frame = pd.read_parquet(cases_path)
    metadata_path = repo_root / "data/interim/pv_glass_cases_with_metadata.csv"
    if metadata_path.exists():
        case_metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False).fillna("")
        cases_frame = cases_frame.merge(case_metadata[["product_id", "stage_hint"]], on="product_id", how="left")
        cases_frame["stage_hint"] = cases_frame["stage_hint"].fillna("")
    else:
        cases_frame["stage_hint"] = ""

    process_frame = pd.read_parquet(process_path)
    top_k = int(config.get("process_top_k", 10))
    records, top1_frame = retrieve_process_candidates_from_registry(
        cases_frame=cases_frame,
        process_registry=process_frame,
        top_k=top_k,
    )

    topk_path = output_dir / "process_topk.jsonl"
    top1_path = output_dir / "process_top1.csv"
    write_jsonl(records, topk_path)
    top1_frame.to_csv(top1_path, index=False)

    logger.info(
        "process_candidates_retrieved",
        extra={
            "structured": {
                "topk_path": str(topk_path),
                "top1_path": str(top1_path),
                "top_k": top_k,
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "case_count": len(cases_frame),
            "candidate_records": len(records),
            "top1_nonempty": int(top1_frame["process_name"].astype(str).ne("").sum()) if not top1_frame.empty else 0,
        },
    )


if __name__ == "__main__":
    main()
