from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import load_yaml, require_exists
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.retrieval.dense_training import train_dense_model
from open_match_lca.seed import seed_everything


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
    args = build_parser().parse_args()
    seed_everything(args.seed)
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("04_train_dense", LOGS_DIR, config_path=args.config, seed=args.seed)

    train_products = pd.read_parquet(require_exists(Path(args.train_path)))
    dev_products = pd.read_parquet(require_exists(Path(args.dev_path)))
    corpus = pd.read_parquet(require_exists(Path(args.corpus_path)))

    artifacts = train_dense_model(
        train_products=train_products,
        dev_products=dev_products,
        corpus=corpus,
        encoder_name=args.encoder_name,
        output_dir=args.output_dir,
        index_dir=str(config.get("index_dir", Path(args.output_dir) / "index")),
        batch_size=int(config.get("batch_size", 16)),
        epochs=int(config.get("epochs", 1)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        max_length=int(config.get("max_length", 256)),
        top_k=int(config.get("top_k", 50)),
        logger=logger,
    )
    log_final_metrics(
        logger,
        {
            "model_dir": str(artifacts.model_dir),
            "train_pairs_path": str(artifacts.train_pairs_path),
            "dev_run_path": str(artifacts.dev_run_path),
            "dev_metrics_path": str(artifacts.dev_metrics_path),
            "index_dir": str(artifacts.index_dir),
        },
    )


if __name__ == "__main__":
    main()
