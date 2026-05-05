from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.openlca_hybrid import (
    build_openlca_hybrid_registry,
    build_pv_glass_reference_registry,
    load_openlca_audit_records,
    write_registry_outputs,
)
from open_match_lca.io_utils import ensure_directory, load_yaml
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/full_pv_glass_hybrid.yaml")
    parser.add_argument(
        "--repo_root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    config = load_yaml(repo_root / args.config if not Path(args.config).is_absolute() else Path(args.config))
    logger, _ = setup_run_logger("01f_prepare_openlca_hybrid_assets", LOGS_DIR, config_path=args.config)

    sidecar_paths = config.get("sidecar_paths", {})
    audit_path = repo_root / sidecar_paths.get("openlca_asset_audit", "data/interim/openlca_local_asset_audit.json")
    process_registry_path = repo_root / sidecar_paths.get(
        "openlca_process_registry", "data/processed/openlca_process_registry.parquet"
    )
    method_registry_path = repo_root / sidecar_paths.get(
        "openlca_method_registry", "data/processed/openlca_method_registry.parquet"
    )
    repo_registry_path = repo_root / "data/interim/openlca_repo_registry.csv"
    reference_registry_path = repo_root / sidecar_paths.get(
        "pv_glass_reference_registry", "data/interim/pv_glass_reference_registry.csv"
    )
    standardized_corpus_path = repo_root / "data/interim/pv_glass_process_corpus_standardized.csv"

    audit_frame = load_openlca_audit_records(audit_path)
    repo_registry, process_registry, method_registry = build_openlca_hybrid_registry(
        audit_frame=audit_frame,
        repo_root=repo_root,
        standardized_process_corpus_path=standardized_corpus_path,
    )
    reference_registry = build_pv_glass_reference_registry(repo_root=repo_root)

    write_registry_outputs(
        repo_registry=repo_registry,
        process_registry=process_registry,
        method_registry=method_registry,
        repo_registry_csv=repo_registry_path,
        process_registry_parquet=process_registry_path,
        method_registry_parquet=method_registry_path,
    )
    ensure_directory(reference_registry_path.parent)
    reference_registry.to_csv(reference_registry_path, index=False)

    logger.info(
        "openlca_hybrid_assets_prepared",
        extra={
            "structured": {
                "repo_registry_path": str(repo_registry_path),
                "process_registry_path": str(process_registry_path),
                "method_registry_path": str(method_registry_path),
                "reference_registry_path": str(reference_registry_path),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "repo_rows": len(repo_registry),
            "process_rows": len(process_registry),
            "calculable_process_rows": int(process_registry["calculable_flag"].fillna(False).sum()) if not process_registry.empty else 0,
            "method_rows": len(method_registry),
            "reference_rows": len(reference_registry),
        },
    )


if __name__ == "__main__":
    main()
