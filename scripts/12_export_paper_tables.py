from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import ensure_directory
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.reporting.export_latex import export_latex_table
from open_match_lca.reporting.export_tables import export_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=["csv", "latex", "both"], required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("12_export_paper_tables", LOGS_DIR)
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        frame = pd.read_json(path, typ="series")
        row = frame.to_dict()
        row["metric_file"] = path.name
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise FileNotFoundError(f"No metric json files found in {metrics_dir}")
    if args.format in {"csv", "both"}:
        export_table(summary, output_dir, "paper_metrics_summary")
    if args.format in {"latex", "both"}:
        export_latex_table(summary, output_dir, "paper_metrics_summary")
    log_final_metrics(logger, {"table_rows": len(summary)})


if __name__ == "__main__":
    main()
