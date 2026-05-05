from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.external_assets import fetch_external_assets
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_optional", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("00c_fetch_external_assets", LOGS_DIR)
    frame = fetch_external_assets(force=args.force, include_optional=not args.skip_optional)
    status_counts = frame["status"].value_counts().to_dict() if not frame.empty else {}
    logger.info("external_assets_checked", extra={"structured": {"status_counts": status_counts}})
    log_final_metrics(logger, status_counts)


if __name__ == "__main__":
    main()
