from __future__ import annotations

import argparse

from open_match_lca.constants import LOGS_DIR
from open_match_lca.logging_utils import setup_run_logger
from open_match_lca.regression.predict_regression import raise_predict_regression_not_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_ckpt", required=True)
    parser.add_argument("--reranker_ckpt", required=False)
    parser.add_argument("--regressor_ckpt", required=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    _ = build_parser().parse_args()
    setup_run_logger("08_predict_all", LOGS_DIR)
    raise_predict_regression_not_ready()


if __name__ == "__main__":
    main()
