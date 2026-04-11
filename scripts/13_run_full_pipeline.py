from __future__ import annotations

import argparse

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import dump_json, load_yaml
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.pipeline.orchestration import build_pipeline_manifest, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.exp_config)
    logger, _ = setup_run_logger("13_run_full_pipeline", LOGS_DIR, config_path=args.exp_config, seed=args.seed)
    if args.dry_run:
        manifest = build_pipeline_manifest(config, args.seed, args.output_dir)
        dump_json(manifest, f"{args.output_dir}/pipeline_manifest.json")
    else:
        manifest = run_pipeline(config, args.seed, args.output_dir, dry_run=False)
    log_final_metrics(
        logger,
        {"dry_run": bool(args.dry_run), "step_count": len(manifest["steps"]), "output_dir": args.output_dir},
    )


if __name__ == "__main__":
    main()
