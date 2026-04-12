from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.eval.eval_regression import compute_regression_metrics
from open_match_lca.constants import LOGS_DIR
from open_match_lca.eval.eval_retrieval import compute_retrieval_metrics
from open_match_lca.eval.eval_uncertainty import evaluate_uncertainty
from open_match_lca.io_utils import dump_json, ensure_directory, load_yaml, read_jsonl
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def _default_eval_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count // 2))


def _resolve_eval_workers(config: dict) -> int:
    if "eval_num_workers" in config:
        return max(1, int(config["eval_num_workers"]))
    if "metric_num_workers" in config:
        return max(1, int(config["metric_num_workers"]))
    return _default_eval_workers()


def _evaluate_retrieval_file(pred_file: str) -> tuple[str, dict[str, float]]:
    path = Path(pred_file)
    records = read_jsonl(path)
    metrics = compute_retrieval_metrics(records)
    suffix = path.stem.replace("retrieval_topk_", "")
    return suffix, metrics


def _evaluate_regression_file(pred_file: str, enable_uncertainty: bool) -> tuple[str, dict, dict | None]:
    path = Path(pred_file)
    frame = pd.read_parquet(path)
    if "y_true" not in frame.columns:
        raise RuntimeError(f"Regression prediction file is missing y_true column: {path}")
    eval_frame = frame.dropna(subset=["y_true", "pred_factor_value"]).reset_index(drop=True)
    if eval_frame.empty:
        metrics = {
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "spearman": None,
            "evaluated_rows": 0,
        }
    else:
        metrics = compute_regression_metrics(
            eval_frame["y_true"].tolist(),
            eval_frame["pred_factor_value"].tolist(),
        )
        metrics["evaluated_rows"] = int(len(eval_frame))

    uncertainty_metrics = None
    required_uncertainty = {"y_true", "lower_conformal", "upper_conformal", "confidence", "error"}
    if enable_uncertainty and required_uncertainty.issubset(frame.columns):
        uncertainty_input = frame.rename(columns={"lower_conformal": "lower", "upper_conformal": "upper"}).copy()
        if "correct" not in uncertainty_input.columns:
            uncertainty_input["correct"] = (
                (uncertainty_input["y_true"] >= uncertainty_input["lower"])
                & (uncertainty_input["y_true"] <= uncertainty_input["upper"])
            ).astype(float)
        uncertainty_metrics = evaluate_uncertainty(uncertainty_input)
    suffix = path.stem.replace("regression_preds_", "")
    return suffix, metrics, uncertainty_metrics


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("09_evaluate_all", LOGS_DIR, config_path=args.config)
    pred_root = Path(args.pred_dir)
    out_root = Path(args.output_dir)
    ensure_directory(out_root)
    eval_workers = _resolve_eval_workers(config)
    retrieval_metrics_count = 0
    regression_metrics_count = 0
    uncertainty_metrics_count = 0
    retrieval_files = [str(path) for path in sorted(pred_root.glob("retrieval_topk_*.jsonl"))]
    regression_files = [str(path) for path in sorted(pred_root.glob("regression_preds_*.parquet"))]
    enable_uncertainty = bool(config.get("whether_uncertainty", True))

    if retrieval_files:
        if eval_workers == 1 or len(retrieval_files) == 1:
            retrieval_results = [_evaluate_retrieval_file(path) for path in retrieval_files]
        else:
            with ProcessPoolExecutor(max_workers=min(eval_workers, len(retrieval_files))) as executor:
                retrieval_results = list(executor.map(_evaluate_retrieval_file, retrieval_files))
        for suffix, metrics in retrieval_results:
            output_path = out_root / f"retrieval_metrics_{suffix}.json"
            dump_json(metrics, output_path)
            retrieval_metrics_count += 1

    if regression_files:
        regression_inputs = [(path, enable_uncertainty) for path in regression_files]
        if eval_workers == 1 or len(regression_files) == 1:
            regression_results = [_evaluate_regression_file(path, enabled) for path, enabled in regression_inputs]
        else:
            with ProcessPoolExecutor(max_workers=min(eval_workers, len(regression_files))) as executor:
                regression_results = list(
                    executor.map(
                        _evaluate_regression_file,
                        [path for path, _ in regression_inputs],
                        [enabled for _, enabled in regression_inputs],
                    )
                )
        for suffix, metrics, uncertainty_metrics in regression_results:
            output_path = out_root / f"regression_metrics_{suffix}.json"
            dump_json(metrics, output_path)
            regression_metrics_count += 1
            if uncertainty_metrics is not None:
                uncertainty_path = out_root / f"uncertainty_metrics_{suffix}.json"
                dump_json(uncertainty_metrics, uncertainty_path)
                uncertainty_metrics_count += 1
    log_final_metrics(
        logger,
        {
            "retrieval_metric_files": retrieval_metrics_count,
            "regression_metric_files": regression_metrics_count,
            "uncertainty_metric_files": uncertainty_metrics_count,
            "eval_num_workers": eval_workers,
        },
    )


if __name__ == "__main__":
    main()
