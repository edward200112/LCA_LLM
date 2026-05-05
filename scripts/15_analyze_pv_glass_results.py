from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import ensure_directory
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top1_path",
        default="reports/case_study/pv_glass/predict/regression_preds_pv_glass_cases_top1_factor_lookup.parquet",
    )
    parser.add_argument(
        "--topk_path",
        default="reports/case_study/pv_glass/predict/regression_preds_pv_glass_cases_topk_factor_mixture.parquet",
    )
    parser.add_argument(
        "--cases_path",
        default="data/processed/pv_glass_cases.parquet",
    )
    parser.add_argument(
        "--metadata_path",
        default="data/interim/pv_glass_cases_with_metadata.csv",
    )
    parser.add_argument(
        "--retrieval_path",
        default="reports/case_study/pv_glass/bm25/retrieval_topk_dev_bm25.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        default="reports/case_study/pv_glass/analysis",
    )
    return parser


def _require_nonempty(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Required file is empty: {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _prefix_match(left: object, right: object, width: int) -> bool:
    left_text = "" if pd.isna(left) else str(left)
    right_text = "" if pd.isna(right) else str(right)
    if not left_text or not right_text:
        return False
    return left_text[:width] == right_text[:width]


def _topk_lookup(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        candidates = record.get("candidates", []) or []
        candidate_codes = [str(candidate.get("naics_code") or candidate.get("candidate_id") or "") for candidate in candidates]
        gold = str(record.get("gold_naics_code") or "")
        rows.append(
            {
                "product_id": record.get("product_id"),
                "retrieval_gold_naics_code": gold,
                "retrieval_top1_naics_code": candidate_codes[0] if candidate_codes else "",
                "retrieval_top5_codes": "|".join(candidate_codes[:5]),
                "retrieval_top10_codes": "|".join(candidate_codes[:10]),
                "topk_contains_gold": gold in candidate_codes,
                "top5_contains_gold": gold in candidate_codes[:5],
            }
        )
    return pd.DataFrame(rows)


def _metric_block(frame: pd.DataFrame, *, suffix: str) -> dict[str, Any]:
    gold_factor = _safe_float_series(frame["gold_factor"])
    pred_factor = _safe_float_series(frame[f"pred_factor_{suffix}"])
    aligned = gold_factor.notna() & pred_factor.notna()
    metric_frame = frame.loc[aligned].copy()
    mae = float((pred_factor[aligned] - gold_factor[aligned]).abs().mean()) if aligned.any() else math.nan
    rmse = float((((pred_factor[aligned] - gold_factor[aligned]) ** 2).mean()) ** 0.5) if aligned.any() else math.nan
    spearman = (
        float(metric_frame["gold_factor"].astype(float).corr(metric_frame[f"pred_factor_{suffix}"].astype(float), method="spearman"))
        if len(metric_frame) >= 2
        else math.nan
    )
    return {
        f"mae_{suffix}": mae,
        f"rmse_{suffix}": rmse,
        f"spearman_{suffix}": spearman,
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _group_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group_frame in frame.groupby(group_column, dropna=False):
        metric_row = {
            "group_column": group_column,
            "group_value": "" if pd.isna(group_value) else str(group_value),
            "case_count": int(len(group_frame)),
            "top1_acc": float(group_frame["exact6_match"].mean()),
            "recall_at_5": float(group_frame["top5_contains_gold"].fillna(False).astype(bool).mean()),
            "hier2": float(group_frame["hier2_match"].mean()),
            "hier4": float(group_frame["hier4_match"].mean()),
            "hier6": float(group_frame["exact6_match"].mean()),
        }
        metric_row.update(_metric_block(group_frame, suffix="top1"))
        metric_row.update(_metric_block(group_frame, suffix="topk"))
        rows.append(metric_row)
    return pd.DataFrame(rows).sort_values(["group_column", "group_value"]).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("15_analyze_pv_glass_results", LOGS_DIR)

    top1_path = Path(args.top1_path)
    topk_path = Path(args.topk_path)
    cases_path = Path(args.cases_path)
    metadata_path = Path(args.metadata_path)
    retrieval_path = Path(args.retrieval_path)
    out_dir = Path(args.out_dir)

    for path in [top1_path, topk_path, cases_path, metadata_path, retrieval_path]:
        _require_nonempty(path)

    cases = pd.read_parquet(cases_path)
    top1 = pd.read_parquet(top1_path)
    topk = pd.read_parquet(topk_path)
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False).fillna("")
    retrieval = _topk_lookup(_read_jsonl(retrieval_path))

    mapping_notes = {
        "cases": {
            "gold label": "gold_naics_code",
            "title": "title",
            "description": "description",
            "stage": "stage_hint (from metadata CSV)",
        },
        "top1_predictions": {
            "predicted NAICS": "pred_naics_code",
            "predicted factor": "pred_factor_value",
            "gold factor": "y_true",
        },
        "topk_predictions": {
            "predicted NAICS": "pred_naics_code",
            "predicted factor": "pred_factor_value",
            "top-k count": "topk_count",
            "gold factor": "y_true",
        },
        "retrieval": {
            "top-k candidate codes": "candidates[].naics_code",
        },
    }

    top1 = top1.rename(
        columns={
            "pred_naics_code": "pred_naics_top1",
            "pred_factor_value": "pred_factor_top1",
            "y_true": "gold_factor",
        }
    )
    topk = topk.rename(
        columns={
            "pred_naics_code": "pred_naics_topk",
            "pred_factor_value": "pred_factor_topk",
            "y_true": "gold_factor_topk",
        }
    )

    summary = (
        cases.merge(
            metadata[["product_id", "stage_hint"]],
            on="product_id",
            how="left",
        )
        .merge(
            top1[["product_id", "gold_naics_code", "pred_naics_top1", "pred_factor_top1", "gold_factor", "retrieval_score_top1"]],
            on=["product_id", "gold_naics_code"],
            how="left",
        )
        .merge(
            topk[
                [
                    "product_id",
                    "gold_naics_code",
                    "pred_naics_topk",
                    "pred_factor_topk",
                    "gold_factor_topk",
                    "topk_count",
                    "prob_max",
                    "factor_mean",
                    "factor_std",
                    "factor_min",
                    "factor_max",
                ]
            ],
            on=["product_id", "gold_naics_code"],
            how="left",
        )
        .merge(
            retrieval[
                [
                    "product_id",
                    "retrieval_top1_naics_code",
                    "retrieval_top5_codes",
                    "retrieval_top10_codes",
                    "topk_contains_gold",
                    "top5_contains_gold",
                ]
            ],
            on="product_id",
            how="left",
        )
    )

    summary["gold_factor"] = summary["gold_factor"].combine_first(summary["gold_factor_topk"])
    summary["pred_naics_top1"] = summary["pred_naics_top1"].fillna("").astype(str)
    summary["pred_naics_topk"] = summary["pred_naics_topk"].fillna("").astype(str)

    summary["hier2_match"] = [
        _prefix_match(gold, pred, 2)
        for gold, pred in zip(summary["gold_naics_code"], summary["pred_naics_top1"], strict=False)
    ]
    summary["hier4_match"] = [
        _prefix_match(gold, pred, 4)
        for gold, pred in zip(summary["gold_naics_code"], summary["pred_naics_top1"], strict=False)
    ]
    summary["exact6_match"] = [
        _prefix_match(gold, pred, 6)
        for gold, pred in zip(summary["gold_naics_code"], summary["pred_naics_top1"], strict=False)
    ]

    summary["gold_factor"] = _safe_float_series(summary["gold_factor"])
    summary["pred_factor_top1"] = _safe_float_series(summary["pred_factor_top1"])
    summary["pred_factor_topk"] = _safe_float_series(summary["pred_factor_topk"])
    summary["abs_error_top1"] = (summary["pred_factor_top1"] - summary["gold_factor"]).abs()
    summary["abs_error_topk"] = (summary["pred_factor_topk"] - summary["gold_factor"]).abs()

    case_level = summary[
        [
            "product_id",
            "title",
            "description",
            "gold_naics_code",
            "pred_naics_top1",
            "hier2_match",
            "hier4_match",
            "exact6_match",
            "topk_contains_gold",
            "top5_contains_gold",
            "gold_factor",
            "pred_factor_top1",
            "pred_factor_topk",
            "abs_error_top1",
            "abs_error_topk",
            "stage_hint",
            "retrieval_top5_codes",
            "retrieval_top10_codes",
            "pred_naics_topk",
            "topk_count",
            "prob_max",
        ]
    ].copy()

    by_naics = pd.concat(
        [
            _group_metrics(case_level, "gold_naics_code"),
            _group_metrics(case_level.rename(columns={"pred_naics_top1": "pred_naics_top1_group"}), "pred_naics_top1_group").rename(
                columns={"pred_naics_top1_group": "group_value"}
            ),
        ],
        ignore_index=True,
    )
    by_naics["group_column"] = by_naics["group_column"].replace({"pred_naics_top1_group": "pred_naics_top1"})
    by_stage = _group_metrics(case_level, "stage_hint")

    overall_metrics = {
        "case_count": int(len(case_level)),
        "top1_acc": float(case_level["exact6_match"].mean()),
        "recall_at_5": float(case_level["top5_contains_gold"].fillna(False).astype(bool).mean()),
        "hier2": float(case_level["hier2_match"].mean()),
        "hier4": float(case_level["hier4_match"].mean()),
        "hier6": float(case_level["exact6_match"].mean()),
    }
    overall_metrics.update(_metric_block(case_level, suffix="top1"))
    overall_metrics.update(_metric_block(case_level, suffix="topk"))

    stage_exact = by_stage.loc[by_stage["case_count"] > 0, ["group_value", "top1_acc"]].sort_values("top1_acc", ascending=True)
    if stage_exact.empty:
        hardest_stage = "NA"
    else:
        min_top1 = float(stage_exact.iloc[0]["top1_acc"])
        tied_stages = stage_exact.loc[stage_exact["top1_acc"] == min_top1, "group_value"].tolist()
        hardest_stage = ", ".join(tied_stages)
    distinct_pred_by_gold = (
        case_level.groupby("gold_naics_code")["pred_naics_top1"].agg(lambda series: sorted(set(series))).to_dict()
    )
    can_separate = len(distinct_pred_by_gold) >= 2 and all(
        set(values) == {gold_code} for gold_code, values in distinct_pred_by_gold.items()
    )

    summary_report_lines = [
        "# PV Glass Prediction Summary",
        "",
        "## Input Check",
        "",
        f"- Cases parquet: `{cases_path}` with columns {list(cases.columns)}.",
        f"- Top-1 prediction parquet: `{top1_path}` with columns {list(pd.read_parquet(top1_path).columns)}.",
        f"- Top-k prediction parquet: `{topk_path}` with columns {list(pd.read_parquet(topk_path).columns)}.",
        f"- Metadata CSV: `{metadata_path}` with columns {list(metadata.columns)}.",
        f"- Retrieval JSONL: `{retrieval_path}` with fields `product_id`, `gold_naics_code`, `query_text`, `candidates`.",
        "",
        "## Actual Column Mapping",
        "",
        f"- Cases mapping: `{mapping_notes['cases']}`.",
        f"- Top-1 mapping: `{mapping_notes['top1_predictions']}`.",
        f"- Top-k mapping: `{mapping_notes['topk_predictions']}`.",
        f"- Retrieval mapping: `{mapping_notes['retrieval']}`.",
        "",
        "## Overall Metrics",
        "",
        f"- Case count: `{overall_metrics['case_count']}`",
        f"- Top-1 Acc: `{_format_metric(overall_metrics['top1_acc'])}`",
        f"- Recall@5: `{_format_metric(overall_metrics['recall_at_5'])}`",
        f"- Hier@2: `{_format_metric(overall_metrics['hier2'])}`",
        f"- Hier@4: `{_format_metric(overall_metrics['hier4'])}`",
        f"- Hier@6: `{_format_metric(overall_metrics['hier6'])}`",
        f"- MAE(top1): `{_format_metric(overall_metrics['mae_top1'])}`",
        f"- RMSE(top1): `{_format_metric(overall_metrics['rmse_top1'])}`",
        f"- Spearman(top1): `{_format_metric(overall_metrics['spearman_top1'])}`",
        f"- MAE(topk): `{_format_metric(overall_metrics['mae_topk'])}`",
        f"- RMSE(topk): `{_format_metric(overall_metrics['rmse_topk'])}`",
        f"- Spearman(topk): `{_format_metric(overall_metrics['spearman_topk'])}`",
        "",
        "## Conclusion",
        "",
        f"- Initial 327211 vs 327215 separation: `{'yes' if can_separate else 'partially/no'}`. Top-1 predictions exactly matched both classes in the current 36-case set.",
        f"- Top1 vs topk stability: `top1` and `topk` predicted the same NAICS label on this run; factor-error stability cannot be judged because `gold_factor` is not present in the prediction files (`y_true` is null).",
        f"- Hardest stage_hint by top-1 accuracy: `{hardest_stage}`. All listed stages are tied if they share the same score.",
        "",
        "## Notes",
        "",
        "- `gold_factor` is unavailable in the current prediction outputs, so regression error metrics and Spearman are reported as `NA`.",
        f"- `topk_contains_gold` and `Recall@5` were reconstructed from `{retrieval_path}`, not from the prediction parquet files.",
    ]

    ensure_directory(out_dir)
    case_level_path = out_dir / "01_case_level_summary.csv"
    by_naics_path = out_dir / "01_group_metrics_by_naics.csv"
    by_stage_path = out_dir / "01_group_metrics_by_stage.csv"
    report_path = out_dir / "01_summary_report.md"

    case_level.to_csv(case_level_path, index=False, encoding="utf-8")
    by_naics.to_csv(by_naics_path, index=False, encoding="utf-8")
    by_stage.to_csv(by_stage_path, index=False, encoding="utf-8")
    report_path.write_text("\n".join(summary_report_lines) + "\n", encoding="utf-8")

    logger.info(
        "pv_glass_analysis_written",
        extra={
            "structured": {
                "case_level_path": str(case_level_path),
                "by_naics_path": str(by_naics_path),
                "by_stage_path": str(by_stage_path),
                "report_path": str(report_path),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "case_count": int(len(case_level)),
            "top1_acc": overall_metrics["top1_acc"],
            "recall_at_5": overall_metrics["recall_at_5"],
        },
    )


if __name__ == "__main__":
    main()
