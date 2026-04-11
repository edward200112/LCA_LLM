from __future__ import annotations

import argparse

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import load_yaml
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.retrieval.rerank_cross_encoder import (
    load_pair_frame,
    train_cross_encoder_reranker,
)
from open_match_lca.seed import seed_everything


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
    args = build_parser().parse_args()
    seed_everything(args.seed)
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("06_train_reranker", LOGS_DIR, config_path=args.config, seed=args.seed)
    train_pairs = load_pair_frame(args.train_pairs)
    dev_pairs = load_pair_frame(args.dev_pairs)
    artifacts = train_cross_encoder_reranker(
        train_pairs=train_pairs,
        dev_pairs=dev_pairs,
        base_model=args.base_model,
        output_dir=args.output_dir,
        batch_size=int(config.get("batch_size", 8)),
        epochs=int(config.get("epochs", 1)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        max_length=int(config.get("max_length", 256)),
    )
    log_final_metrics(
        logger,
        {
            "model_dir": str(artifacts.model_dir),
            "dev_pair_scores_path": str(artifacts.dev_pair_scores_path),
            "train_pair_count": artifacts.train_pair_count,
            "dev_pair_count": artifacts.dev_pair_count,
        },
    )


if __name__ == "__main__":
    main()
