from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.openlca_hybrid import (
    audit_openlca_local_assets,
    summarize_audit_for_terminal,
    write_openlca_audit_outputs,
)
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument("--json_out", default="data/interim/openlca_local_asset_audit.json")
    parser.add_argument("--csv_out", default="data/interim/openlca_local_asset_audit.csv")
    parser.add_argument("--md_out", default="data/interim/openlca_local_asset_audit.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    logger, _ = setup_run_logger("00f_audit_openlca_local_assets", LOGS_DIR)

    audit_frame = audit_openlca_local_assets(repo_root=repo_root)
    write_openlca_audit_outputs(
        audit_frame=audit_frame,
        json_path=repo_root / args.json_out,
        csv_path=repo_root / args.csv_out,
        md_path=repo_root / args.md_out,
    )
    summary = summarize_audit_for_terminal(audit_frame)
    print(summary)
    logger.info(
        "openlca_asset_audit_written",
        extra={
            "structured": {
                "summary": summary,
                "json_out": str(repo_root / args.json_out),
                "csv_out": str(repo_root / args.csv_out),
                "md_out": str(repo_root / args.md_out),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "rows": len(audit_frame),
            "importable_assets": int(audit_frame["likely_importable_to_openlca"].fillna(False).sum()) if not audit_frame.empty else 0,
        },
    )


if __name__ == "__main__":
    main()
