from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import dump_json, load_yaml
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.pipeline.orchestration import (
    DEFAULT_ABLATIONS,
    apply_ablation,
    build_pipeline_manifest,
    materialize_ablation_configs,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ablation", default="all")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("10_run_ablation", LOGS_DIR, config_path=args.exp_config, seed=args.seed)
    base_config = load_yaml(args.exp_config)
    ablations = DEFAULT_ABLATIONS if args.ablation == "all" else [args.ablation]
    config_paths = materialize_ablation_configs(args.exp_config, f"{args.output_dir}/configs", ablations)
    manifests = []
    for ablation_name, config_path in zip(ablations, config_paths, strict=False):
        config = apply_ablation(base_config, ablation_name)
        if args.dry_run:
            manifest = build_pipeline_manifest(config, args.seed, f"{args.output_dir}/{ablation_name}")
            dump_json(manifest, f"{args.output_dir}/{ablation_name}/pipeline_manifest.json")
        else:
            manifest = run_pipeline(config, args.seed, f"{args.output_dir}/{ablation_name}", dry_run=False)
        manifests.append({"ablation": ablation_name, "config_path": str(config_path), "manifest": manifest})
    dump_json({"runs": manifests}, f"{args.output_dir}/ablation_manifest.json")
    log_final_metrics(logger, {"ablation_count": len(ablations), "dry_run": bool(args.dry_run)})


if __name__ == "__main__":
    main()
