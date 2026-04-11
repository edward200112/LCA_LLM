from __future__ import annotations

import argparse

from open_match_lca.constants import LOGS_DIR
from open_match_lca.logging_utils import setup_run_logger
from open_match_lca.regression.train_lgbm_regressor import raise_lgbm_not_ready
from open_match_lca.regression.train_xgb_regressor import raise_xgb_not_ready


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
    setup_run_logger("07_train_regressor", LOGS_DIR)
    if args.model == "lgbm":
        raise_lgbm_not_ready()
    raise_xgb_not_ready()


if __name__ == "__main__":
    main()
