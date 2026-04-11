from __future__ import annotations

import argparse
from pathlib import Path

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import read_jsonl, write_jsonl
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.retrieval.hybrid_rrf import reciprocal_rank_fusion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25_run", required=True)
    parser.add_argument("--dense_run", required=True)
    parser.add_argument("--fusion", choices=["rrf"], required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--output_path", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("05_build_hybrid", LOGS_DIR)
    bm25_run = read_jsonl(args.bm25_run)
    dense_run = read_jsonl(args.dense_run)
    if len(bm25_run) != len(dense_run):
        raise RuntimeError(
            f"bm25_run and dense_run length mismatch: {len(bm25_run)} vs {len(dense_run)}"
        )

    fused = []
    for left, right in zip(bm25_run, dense_run, strict=False):
        fused.append(
            {
                "product_id": left["product_id"],
                "gold_naics_code": left["gold_naics_code"],
                "query_text": left.get("query_text", ""),
                "candidates": reciprocal_rank_fusion(
                    [left.get("candidates", []), right.get("candidates", [])],
                    top_k=args.topk,
                ),
            }
        )
    output_path = Path(args.output_path)
    write_jsonl(fused, output_path)
    log_final_metrics(logger, {"queries": len(fused)})


if __name__ == "__main__":
    main()
