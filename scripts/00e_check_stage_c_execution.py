from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.pv_glass_stage_workflows import write_stage_c_commands, write_stage_c_execution_check
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def main() -> None:
    _ = build_parser().parse_args()
    logger, _ = setup_run_logger("00e_check_stage_c_execution", LOGS_DIR)
    check_path = write_stage_c_execution_check()
    commands_path = write_stage_c_commands()
    logger.info(
        "stage_c_reports_written",
        extra={
            "structured": {
                "execution_check_path": str(check_path),
                "commands_path": str(commands_path),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "execution_check_path": str(check_path),
            "commands_path": str(commands_path),
        },
    )


if __name__ == "__main__":
    main()
