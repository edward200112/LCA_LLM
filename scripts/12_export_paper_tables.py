from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import ensure_directory
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.reporting.export_figures import (
    export_bar_chart,
    export_calibration_plot,
    export_histogram,
    export_line_chart,
)
from open_match_lca.reporting.export_latex import export_latex_table
from open_match_lca.reporting.export_tables import export_table
from open_match_lca.uncertainty.abstention import risk_coverage_curve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=["csv", "latex", "both"], required=True)
    parser.add_argument("--pred_dir", required=False)
    parser.add_argument("--products_path", required=False)
    return parser


def _load_metric_rows(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        row = dict(payload)
        row["metric_file"] = path.name
        row["metric_type"] = path.stem.split("_metrics_")[0]
        row["split_model"] = path.stem.split("_metrics_")[-1]
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No metric json files found in {metrics_dir}")
    return frame


def _write_table_bundle(frame: pd.DataFrame, output_dir: Path, stem: str, fmt: str) -> None:
    if frame.empty:
        return
    if fmt in {"csv", "both"}:
        export_table(frame, output_dir, stem)
    if fmt in {"latex", "both"}:
        export_latex_table(frame, output_dir, stem)


def _export_figures(pred_dir: Path | None, products_path: Path | None, figure_dir: Path, metric_rows: pd.DataFrame) -> int:
    ensure_directory(figure_dir)
    figure_count = 0
    retrieval_rows = metric_rows.loc[metric_rows["metric_type"] == "retrieval"].copy()
    if not retrieval_rows.empty and "top1_accuracy" in retrieval_rows.columns:
        retrieval_plot = retrieval_rows.loc[:, ["split_model", "top1_accuracy"]].fillna(0.0)
        export_bar_chart(
            retrieval_plot,
            x="split_model",
            y="top1_accuracy",
            output_path=figure_dir / "retrieval_performance_bar_chart.png",
            title="Retrieval Top-1 Accuracy",
        )
        figure_count += 1

    if products_path is not None and products_path.exists():
        products = pd.read_parquet(products_path)
        if "gold_naics_code" in products.columns:
            distribution = (
                products["gold_naics_code"].astype(str).value_counts().head(20).reset_index()
            )
            distribution.columns = ["gold_naics_code", "count"]
            export_bar_chart(
                distribution,
                x="gold_naics_code",
                y="count",
                output_path=figure_dir / "class_distribution.png",
                title="Class Distribution",
            )
            figure_count += 1

    if pred_dir is not None and pred_dir.exists():
        reg_files = sorted(pred_dir.glob("regression_preds_*_lgbm_regressor.parquet"))
        if reg_files:
            frame = pd.read_parquet(reg_files[0])
            if {"confidence", "correct"}.issubset(frame.columns):
                export_calibration_plot(
                    frame,
                    confidence_col="confidence",
                    correctness_col="correct",
                    output_path=figure_dir / "calibration_plot.png",
                    title="Calibration Plot",
                )
                figure_count += 1
            if {"confidence", "error"}.issubset(frame.columns):
                curve = risk_coverage_curve(frame, "confidence", "error")
                export_line_chart(
                    curve,
                    x="coverage",
                    y="retained_risk",
                    output_path=figure_dir / "risk_coverage_curve.png",
                    title="Risk-Coverage Curve",
                )
                figure_count += 1
            if "error" in frame.columns:
                export_histogram(
                    frame["error"].astype(float),
                    output_path=figure_dir / "error_case_histogram.png",
                    title="Error Case Histogram",
                )
                figure_count += 1
    return figure_count


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("12_export_paper_tables", LOGS_DIR)
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    metric_rows = _load_metric_rows(metrics_dir)

    main_results = metric_rows.loc[
        metric_rows["metric_type"].isin(["retrieval", "regression"])
        & ~metric_rows["metric_file"].str.contains("cluster_ood|hierarchical_zero_shot|ablation", regex=True)
    ].reset_index(drop=True)
    ood_results = metric_rows.loc[
        metric_rows["metric_file"].str.contains("cluster_ood|hierarchical_zero_shot", regex=True)
    ].reset_index(drop=True)
    uncertainty_results = metric_rows.loc[metric_rows["metric_type"] == "uncertainty"].reset_index(drop=True)
    ablation_results = metric_rows.loc[
        metric_rows["metric_file"].str.contains(
            "bm25_only|dense_only|hybrid_no_rerank|hybrid_with_rerank|top1_factor_only|topk_factor_mixture|regressor_off|regressor_on|hierarchy_features_off|uncertainty_off|process_extension_off|process_extension_on",
            regex=True,
        )
    ].reset_index(drop=True)

    _write_table_bundle(main_results, output_dir, "main_results_table", args.format)
    _write_table_bundle(ablation_results, output_dir, "ablation_table", args.format)
    _write_table_bundle(ood_results, output_dir, "ood_results_table", args.format)
    _write_table_bundle(uncertainty_results, output_dir, "uncertainty_table", args.format)
    _write_table_bundle(metric_rows, output_dir, "paper_metrics_summary", args.format)

    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    products_path = Path(args.products_path) if args.products_path else None
    figure_count = _export_figures(pred_dir, products_path, output_dir.parent / "figures", metric_rows)
    log_final_metrics(logger, {"table_rows": len(metric_rows), "figure_count": figure_count})


if __name__ == "__main__":
    main()
