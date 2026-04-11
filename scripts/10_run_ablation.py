from __future__ import annotations

import argparse

from open_match_lca.constants import LOGS_DIR
from open_match_lca.logging_utils import setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_run_logger("10_run_ablation", LOGS_DIR, config_path=args.exp_config, seed=args.seed)
    raise RuntimeError(
        "Ablation runner is scaffolded only. Planned ablations: bm25_only, dense_only, "
        "hybrid_no_rerank, hybrid_with_rerank, top1_factor_only, topk_factor_mixture, "
        "regressor_off, regressor_on, hierarchy_features_off, uncertainty_off, "
        "process_extension_off, process_extension_on."
    )


if __name__ == "__main__":
    main()
