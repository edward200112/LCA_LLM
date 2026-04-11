from __future__ import annotations

import argparse

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.download_public_data import PUBLIC_DATASET_MANIFEST, scaffold_download_targets
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["all", *PUBLIC_DATASET_MANIFEST.keys()], required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("00_download_data", LOGS_DIR)
    manifest = scaffold_download_targets(args.target, args.out_dir, overwrite=args.overwrite)
    logger.info("download_targets_scaffolded", extra={"structured": {"manifest": manifest}})
    log_final_metrics(logger, {"targets": len(manifest)})


if __name__ == "__main__":
    main()
