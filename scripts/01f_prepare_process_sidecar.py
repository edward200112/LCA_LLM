from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.process_sidecar import prepare_process_sidecar
from open_match_lca.io_utils import load_yaml
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/process_sidecar.yaml")
    parser.add_argument(
        "--repo_root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("01f_prepare_process_sidecar", LOGS_DIR, config_path=args.config)
    outputs = prepare_process_sidecar(config, repo_root=args.repo_root.resolve(), force=args.force)
    logger.info(
        "process_sidecar_prepared",
        extra={"structured": {"outputs": {key: str(value) for key, value in outputs.items()}}},
    )
    log_final_metrics(logger, {"artifacts": len(outputs)})


if __name__ == "__main__":
    main()
