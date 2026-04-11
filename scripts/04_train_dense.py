from __future__ import annotations

import argparse

from open_match_lca.constants import LOGS_DIR
from open_match_lca.logging_utils import setup_run_logger
from open_match_lca.retrieval.dense_retriever import raise_dense_not_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    _ = build_parser().parse_args()
    setup_run_logger("04_train_dense", LOGS_DIR)
    raise_dense_not_ready()


if __name__ == "__main__":
    main()
