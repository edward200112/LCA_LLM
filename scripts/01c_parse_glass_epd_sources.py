from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.pv_glass_extension import build_glass_factor_registry, build_pv_glass_process_corpus
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("01c_parse_glass_epd_sources", LOGS_DIR)
    factor_registry_path = build_glass_factor_registry(force=args.force)
    process_corpus_path = build_pv_glass_process_corpus(force=args.force)
    logger.info(
        "glass_sources_parsed",
        extra={
            "structured": {
                "factor_registry_path": str(factor_registry_path),
                "process_corpus_path": str(process_corpus_path),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "factor_registry_path": str(factor_registry_path),
            "process_corpus_path": str(process_corpus_path),
        },
    )


if __name__ == "__main__":
    main()
