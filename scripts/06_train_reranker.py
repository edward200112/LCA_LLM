from __future__ import annotations

import argparse

from open_match_lca.constants import LOGS_DIR
from open_match_lca.logging_utils import setup_run_logger
from open_match_lca.retrieval.rerank_cross_encoder import raise_reranker_not_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--dev_pairs", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    _ = build_parser().parse_args()
    setup_run_logger("06_train_reranker", LOGS_DIR)
    raise_reranker_not_ready()


if __name__ == "__main__":
    main()
