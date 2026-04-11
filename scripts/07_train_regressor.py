from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import load_yaml, require_exists
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.regression.train_lgbm_regressor import train_lgbm_quantile_regressor
from open_match_lca.regression.train_xgb_regressor import raise_xgb_not_ready
from open_match_lca.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--dev_features", required=True)
    parser.add_argument("--model", choices=["lgbm", "xgb"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("07_train_regressor", LOGS_DIR, config_path=args.config, seed=args.seed)
    if args.model == "lgbm":
        train_frame = pd.read_parquet(require_exists(Path(args.train_features)))
        dev_frame = pd.read_parquet(require_exists(Path(args.dev_features)))
        artifacts = train_lgbm_quantile_regressor(
            train_frame=train_frame,
            dev_frame=dev_frame,
            output_dir=args.output_dir,
            quantiles=tuple(config.get("quantiles", [0.1, 0.5, 0.9])),
            top_k=int(config.get("top_k", 5)),
            pca_dim=int(config.get("pca_dim", 64)),
            seed=args.seed,
            logger=logger,
            use_hierarchy_features=bool(config.get("use_hierarchy_features", True)),
        )
        log_final_metrics(
            logger,
            {
                "bundle_path": str(artifacts.bundle_path),
                "dev_predictions_path": str(artifacts.dev_predictions_path),
                "dev_metrics_path": str(artifacts.dev_metrics_path),
            },
        )
        return
    raise_xgb_not_ready()


if __name__ == "__main__":
    main()
