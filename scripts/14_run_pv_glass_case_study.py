from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.pv_glass_stage_workflows import run_pv_glass_case_study, write_stage_c_run_plan
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/full_pv_glass.yaml")
    parser.add_argument("--mode", choices=["plan", "smoke", "predict", "process"], default="smoke")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("14_run_pv_glass_case_study", LOGS_DIR, config_path=args.config, seed=args.seed)
    if args.mode == "plan":
        report_path = write_stage_c_run_plan(force=args.force)
        logger.info("pv_glass_case_study_plan_written", extra={"structured": {"report_path": str(report_path)}})
        log_final_metrics(logger, {"mode": args.mode, "report_path": str(report_path)})
        return
    outputs = run_pv_glass_case_study(config_path=args.config, mode=args.mode, seed=args.seed)
    logger.info(
        "pv_glass_case_study_completed",
        extra={"structured": {"mode": args.mode, "outputs": {key: str(value) for key, value in outputs.items()}}},
    )
    log_final_metrics(logger, {"mode": args.mode, "output_count": len(outputs)})


if __name__ == "__main__":
    main()
