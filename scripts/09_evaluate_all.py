from __future__ import annotations

import argparse
from pathlib import Path

from open_match_lca.constants import LOGS_DIR
from open_match_lca.eval.eval_retrieval import compute_retrieval_metrics
from open_match_lca.io_utils import dump_json, read_jsonl
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("09_evaluate_all", LOGS_DIR, config_path=args.config)
    pred_root = Path(args.pred_dir)
    out_root = Path(args.output_dir)
    metrics_count = 0
    for pred_file in sorted(pred_root.glob("retrieval_topk_*.jsonl")):
        records = read_jsonl(pred_file)
        metrics = compute_retrieval_metrics(records)
        output_path = out_root / f"retrieval_metrics_{pred_file.stem.replace('retrieval_topk_', '')}.json"
        dump_json(metrics, output_path)
        metrics_count += 1
    log_final_metrics(logger, {"retrieval_metric_files": metrics_count})


if __name__ == "__main__":
    main()
