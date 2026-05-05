from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.openlca_hybrid import run_openlca_calculations
from open_match_lca.io_utils import ensure_directory, load_yaml
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue_path",
        default="reports/case_study/pv_glass/process_calc/calculation_queue.csv",
    )
    parser.add_argument("--config", default="configs/exp/full_pv_glass_hybrid.yaml")
    parser.add_argument("--output_dir", default="reports/case_study/pv_glass/process_calc")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_yaml(repo_root / args.config if not Path(args.config).is_absolute() else Path(args.config))
    logger, _ = setup_run_logger("08d_run_openlca_hybrid_calc", LOGS_DIR, config_path=args.config)

    queue_path = repo_root / args.queue_path if not Path(args.queue_path).is_absolute() else Path(args.queue_path)
    output_dir = repo_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    ensure_directory(output_dir)
    queue_frame = pd.read_csv(queue_path, keep_default_na=False).fillna("")

    openlca_config = config.get("openlca", {})
    method_name = str(openlca_config.get("lcia_method_name") or "").strip()
    if not method_name:
        failures = queue_frame.loc[queue_frame["calc_status"].astype(str) == "queued", ["case_id", "process_uuid", "process_name"]].copy()
        failures["failure_reason"] = "missing_lcia_method_name_config"
        failures.to_csv(output_dir / "openlca_calc_failures.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "openlca_impacts_raw.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "openlca_impacts_normalized.csv", index=False)
        logger.info(
            "openlca_calc_skipped",
            extra={"structured": {"reason": "missing_lcia_method_name_config", "output_dir": str(output_dir)}},
        )
        log_final_metrics(logger, {"queued_rows": int(queue_frame["calc_status"].eq("queued").sum()), "failures": len(failures)})
        return

    raw_frame, normalized_frame, failure_frame = run_openlca_calculations(
        queue_frame=queue_frame,
        method_name=method_name,
        output_dir=output_dir,
        port=int(openlca_config.get("port", 8080)),
        timeout_sec=int(openlca_config.get("timeout_sec", 120)),
    )
    logger.info(
        "openlca_calc_finished",
        extra={
            "structured": {
                "output_dir": str(output_dir),
                "raw_rows": len(raw_frame),
                "normalized_rows": len(normalized_frame),
                "failure_rows": len(failure_frame),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "queued_rows": int(queue_frame["calc_status"].eq("queued").sum()) if not queue_frame.empty else 0,
            "raw_rows": len(raw_frame),
            "normalized_rows": len(normalized_frame),
            "failure_rows": len(failure_frame),
        },
    )


if __name__ == "__main__":
    main()
