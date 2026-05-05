from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.repo_audit import write_repo_state_reports
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return parser


def main() -> None:
    _ = build_parser().parse_args()
    logger, _ = setup_run_logger("00b_audit_repo_and_data", LOGS_DIR)
    report_path, inventory_path = write_repo_state_reports()
    logger.info(
        "repo_audit_written",
        extra={
            "structured": {
                "report_path": str(report_path),
                "inventory_path": str(inventory_path),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "report_path": str(report_path),
            "inventory_path": str(inventory_path),
        },
    )


if __name__ == "__main__":
    main()
