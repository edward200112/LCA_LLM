from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.pv_glass_extension import build_pv_glass_cases
from open_match_lca.data.pv_glass_stage_workflows import write_stage_b_report
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("01e_build_pv_glass_cases", LOGS_DIR)
    raw_path, metadata_path = build_pv_glass_cases(force=args.force)
    report_path = write_stage_b_report(force=args.force)
    logger.info(
        "pv_glass_cases_ready",
        extra={
            "structured": {
                "raw_path": str(raw_path),
                "metadata_path": str(metadata_path),
                "report_path": str(report_path),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "raw_path": str(raw_path),
            "metadata_path": str(metadata_path),
            "report_path": str(report_path),
        },
    )


if __name__ == "__main__":
    main()
