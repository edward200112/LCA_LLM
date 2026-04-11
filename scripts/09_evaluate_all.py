from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from open_match_lca.eval.eval_regression import compute_regression_metrics
from open_match_lca.constants import LOGS_DIR
from open_match_lca.eval.eval_retrieval import compute_retrieval_metrics
from open_match_lca.eval.eval_uncertainty import evaluate_uncertainty
from open_match_lca.io_utils import dump_json, load_yaml, read_jsonl
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
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("09_evaluate_all", LOGS_DIR, config_path=args.config)
    pred_root = Path(args.pred_dir)
    out_root = Path(args.output_dir)
    retrieval_metrics_count = 0
    regression_metrics_count = 0
    uncertainty_metrics_count = 0
    for pred_file in sorted(pred_root.glob("retrieval_topk_*.jsonl")):
        records = read_jsonl(pred_file)
        metrics = compute_retrieval_metrics(records)
        output_path = out_root / f"retrieval_metrics_{pred_file.stem.replace('retrieval_topk_', '')}.json"
        dump_json(metrics, output_path)
        retrieval_metrics_count += 1

    for pred_file in sorted(pred_root.glob("regression_preds_*.parquet")):
        frame = pd.read_parquet(pred_file)
        if "y_true" not in frame.columns:
            raise RuntimeError(
                f"Regression prediction file is missing y_true column: {pred_file}"
            )
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
        output_path = out_root / f"regression_metrics_{pred_file.stem.replace('regression_preds_', '')}.json"
        dump_json(metrics, output_path)
        regression_metrics_count += 1
        required_uncertainty = {"y_true", "lower_conformal", "upper_conformal", "confidence", "error"}
        if bool(config.get("whether_uncertainty", True)) and required_uncertainty.issubset(frame.columns):
            uncertainty_input = frame.rename(
                columns={"lower_conformal": "lower", "upper_conformal": "upper"}
            ).copy()
            if "correct" not in uncertainty_input.columns:
                uncertainty_input["correct"] = (
                    (uncertainty_input["y_true"] >= uncertainty_input["lower"])
                    & (uncertainty_input["y_true"] <= uncertainty_input["upper"])
                ).astype(float)
            uncertainty_metrics = evaluate_uncertainty(uncertainty_input)
            uncertainty_path = out_root / f"uncertainty_metrics_{pred_file.stem.replace('regression_preds_', '')}.json"
            dump_json(uncertainty_metrics, uncertainty_path)
            uncertainty_metrics_count += 1
    log_final_metrics(
        logger,
        {
            "retrieval_metric_files": retrieval_metrics_count,
            "regression_metric_files": regression_metrics_count,
            "uncertainty_metric_files": uncertainty_metrics_count,
        },
    )


if __name__ == "__main__":
    main()
