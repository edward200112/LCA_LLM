from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.pv_glass_stage_workflows import write_stage_a_report
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("00d_inspect_raw_sources", LOGS_DIR)
    report_path = write_stage_a_report(force=args.force, allow_download=not args.skip_download)
    logger.info("stage_a_report_written", extra={"structured": {"report_path": str(report_path)}})
    log_final_metrics(logger, {"report_path": str(report_path)})


if __name__ == "__main__":
    main()
