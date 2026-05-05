from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.pv_glass_extension import prepare_pv_glass_extension
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("01b_prepare_pv_glass_extension", LOGS_DIR)
    outputs = prepare_pv_glass_extension(force=args.force)
    logger.info(
        "pv_glass_extension_prepared",
        extra={"structured": {"outputs": {key: str(value) for key, value in outputs.items()}}},
    )
    log_final_metrics(logger, {"artifacts": len(outputs)})


if __name__ == "__main__":
    main()
