from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.make_splits import make_dataset_splits, write_splits
from open_match_lca.io_utils import require_exists
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_products",
        required=True,
    )
    parser.add_argument(
        "--split_type",
        choices=["random_stratified", "hierarchical_zero_shot", "cluster_ood"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    logger, _ = setup_run_logger("02_make_splits", LOGS_DIR, seed=args.seed)
    products_path = require_exists(Path(args.input_products))
    products = pd.read_parquet(products_path)
    splits = make_dataset_splits(products, args.split_type, args.seed)
    write_splits(splits, args.out_dir, args.split_type)
    sizes = {name: len(frame) for name, frame in splits.items()}
    logger.info("splits_created", extra={"structured": {"sizes": sizes}})
    log_final_metrics(logger, sizes)


if __name__ == "__main__":
    main()
