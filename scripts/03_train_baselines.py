from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import load_yaml, require_exists, write_jsonl
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.retrieval.candidate_generation import (
    bm25_retrieve,
    exact_or_lexical_retrieve,
    tfidf_retrieve,
)
from open_match_lca.retrieval.dense_retriever import raise_dense_not_ready
from open_match_lca.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--model", choices=["exact", "tfidf", "bm25", "dense_zero_shot"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("03_train_baselines", LOGS_DIR, config_path=args.config, seed=args.seed)

    _ = require_exists(Path(args.train_path))
    dev = pd.read_parquet(require_exists(Path(args.dev_path)))
    corpus = pd.read_parquet(require_exists(Path(args.corpus_path)))
    top_k = int(config.get("top_k", 10))

    if args.model == "exact":
        runs = exact_or_lexical_retrieve(dev, corpus, top_k=top_k)
    elif args.model == "tfidf":
        runs = tfidf_retrieve(dev, corpus, top_k=top_k)
    elif args.model == "bm25":
        runs = bm25_retrieve(dev, corpus, top_k=top_k)
    else:
        raise_dense_not_ready()
        return

    output_path = Path(args.output_dir) / f"retrieval_topk_dev_{args.model}.jsonl"
    write_jsonl(runs, output_path)
    logger.info("baseline_run_saved", extra={"structured": {"output_path": str(output_path)}})
    log_final_metrics(logger, {"queries": len(runs)})


if __name__ == "__main__":
    main()
