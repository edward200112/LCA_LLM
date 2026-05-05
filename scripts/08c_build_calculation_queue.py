from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.openlca_hybrid import build_calculation_queue_frame
from open_match_lca.io_utils import ensure_directory, load_yaml, read_jsonl
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_path", default="reports/case_study/pv_glass/process_retrieval/process_topk.jsonl")
    parser.add_argument("--process_registry_path", default="data/processed/openlca_process_registry.parquet")
    parser.add_argument("--cases_path", default="data/processed/pv_glass_cases.parquet")
    parser.add_argument("--case_metadata_path", default="data/interim/pv_glass_cases_with_metadata.csv")
    parser.add_argument("--reference_registry_path", default="data/interim/pv_glass_reference_registry.csv")
    parser.add_argument("--config", default="configs/exp/full_pv_glass_hybrid.yaml")
    parser.add_argument("--output_path", default="reports/case_study/pv_glass/process_calc/calculation_queue.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_yaml(repo_root / args.config if not Path(args.config).is_absolute() else Path(args.config))
    logger, _ = setup_run_logger("08c_build_calculation_queue", LOGS_DIR, config_path=args.config)

    retrieval_path = repo_root / args.retrieval_path if not Path(args.retrieval_path).is_absolute() else Path(args.retrieval_path)
    process_registry_path = (
        repo_root / args.process_registry_path
        if not Path(args.process_registry_path).is_absolute()
        else Path(args.process_registry_path)
    )
    case_metadata_path = (
        repo_root / args.case_metadata_path if not Path(args.case_metadata_path).is_absolute() else Path(args.case_metadata_path)
    )
    reference_registry_path = (
        repo_root / args.reference_registry_path
        if not Path(args.reference_registry_path).is_absolute()
        else Path(args.reference_registry_path)
    )
    output_path = repo_root / args.output_path if not Path(args.output_path).is_absolute() else Path(args.output_path)

    retrieval_records = read_jsonl(retrieval_path)
    process_registry = pd.read_parquet(process_registry_path)
    case_metadata = (
        pd.read_csv(case_metadata_path, dtype=str, keep_default_na=False).fillna("")
        if case_metadata_path.exists()
        else pd.DataFrame()
    )
    reference_registry = (
        pd.read_csv(reference_registry_path, keep_default_na=False).fillna("")
        if reference_registry_path.exists()
        else pd.DataFrame()
    )
    target_unit = config.get("normalization", {}).get("target_unit", "kgCO2e/m2")

    queue_frame = build_calculation_queue_frame(
        retrieval_records=retrieval_records,
        process_registry=process_registry,
        case_metadata_frame=case_metadata,
        reference_registry_frame=reference_registry,
        target_unit=target_unit,
    )

    ensure_directory(output_path.parent)
    queue_frame.to_csv(output_path, index=False)
    logger.info(
        "calculation_queue_written",
        extra={
            "structured": {
                "output_path": str(output_path),
                "queued": int(queue_frame["calc_status"].eq("queued").sum()) if not queue_frame.empty else 0,
                "blocked": int(queue_frame["calc_status"].eq("blocked").sum()) if not queue_frame.empty else 0,
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "rows": len(queue_frame),
            "queued": int(queue_frame["calc_status"].eq("queued").sum()) if not queue_frame.empty else 0,
            "normalization_ready": int(queue_frame["normalization_ready_flag"].fillna(False).sum()) if not queue_frame.empty else 0,
        },
    )


if __name__ == "__main__":
    main()
