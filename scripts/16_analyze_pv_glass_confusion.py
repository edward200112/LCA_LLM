from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import ensure_directory
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case_summary_path",
        default="reports/case_study/pv_glass/analysis/01_case_level_summary.csv",
    )
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
        "--out_dir",
        default="reports/case_study/pv_glass/analysis",
    )
    return parser


def _require_nonempty(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Required file is empty: {path}")


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _prefix_match(left: object, right: object, width: int) -> bool:
    left_text = "" if pd.isna(left) else str(left)
    right_text = "" if pd.isna(right) else str(right)
    if not left_text or not right_text:
        return False
    return left_text[:width] == right_text[:width]


def _load_case_level(case_summary_path: Path, top1_path: Path, topk_path: Path, cases_path: Path) -> tuple[pd.DataFrame, str]:
    if case_summary_path.exists() and case_summary_path.stat().st_size > 0:
        frame = pd.read_csv(case_summary_path, dtype=str, keep_default_na=False).fillna("")
        return frame, "01_case_level_summary.csv"

    for path in [top1_path, topk_path, cases_path]:
        _require_nonempty(path)

    cases = pd.read_parquet(cases_path)
    top1 = pd.read_parquet(top1_path)
    topk = pd.read_parquet(topk_path)

    merged = (
        cases[["product_id", "title", "description", "gold_naics_code"]]
        .merge(
            top1.rename(columns={"pred_naics_code": "pred_naics_top1"})[
                ["product_id", "gold_naics_code", "pred_naics_top1"]
            ],
            on=["product_id", "gold_naics_code"],
            how="left",
        )
        .merge(
            topk.rename(columns={"pred_naics_code": "pred_naics_topk"})[
                ["product_id", "gold_naics_code", "pred_naics_topk"]
            ],
            on=["product_id", "gold_naics_code"],
            how="left",
        )
    )
    merged["hier2_match"] = [
        _prefix_match(gold, pred, 2)
        for gold, pred in zip(merged["gold_naics_code"], merged["pred_naics_top1"], strict=False)
    ]
    merged["hier4_match"] = [
        _prefix_match(gold, pred, 4)
        for gold, pred in zip(merged["gold_naics_code"], merged["pred_naics_top1"], strict=False)
    ]
    merged["exact6_match"] = [
        _prefix_match(gold, pred, 6)
        for gold, pred in zip(merged["gold_naics_code"], merged["pred_naics_top1"], strict=False)
    ]
    merged["topk_contains_gold"] = merged["gold_naics_code"] == merged["pred_naics_topk"]
    merged["stage_hint"] = ""
    return merged.fillna(""), "fallback_rebuild_from_prediction_parquet"


def _write_confusion_heatmap(confusion: pd.DataFrame, out_path: Path) -> bool:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "open_match_lca_mpl"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    labels = confusion.index.tolist()
    matrix = confusion.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(confusion.columns)))
    ax.set_xticklabels(confusion.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted NAICS")
    ax.set_ylabel("Gold NAICS")
    ax.set_title("PV Glass Confusion Matrix")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, int(matrix[row_idx, col_idx]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_directory(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("16_analyze_pv_glass_confusion", LOGS_DIR)

    case_summary_path = Path(args.case_summary_path)
    top1_path = Path(args.top1_path)
    topk_path = Path(args.topk_path)
    cases_path = Path(args.cases_path)
    out_dir = Path(args.out_dir)

    case_level, source_used = _load_case_level(case_summary_path, top1_path, topk_path, cases_path)
    required_columns = ["product_id", "gold_naics_code", "pred_naics_top1"]
    missing_columns = [column for column in required_columns if column not in case_level.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for confusion analysis: {missing_columns}")

    for optional_column in ["hier2_match", "hier4_match", "exact6_match", "topk_contains_gold", "stage_hint", "pred_naics_topk"]:
        if optional_column not in case_level.columns:
            case_level[optional_column] = ""

    case_level["gold_naics_code"] = case_level["gold_naics_code"].astype(str)
    case_level["pred_naics_top1"] = case_level["pred_naics_top1"].astype(str)
    case_level["hier2_match"] = _to_bool(case_level["hier2_match"])
    case_level["hier4_match"] = _to_bool(case_level["hier4_match"])
    case_level["exact6_match"] = _to_bool(case_level["exact6_match"])
    case_level["topk_contains_gold"] = _to_bool(case_level["topk_contains_gold"])
    case_level["stage_hint"] = case_level["stage_hint"].astype(str).replace({"": "unknown"})

    confusion = (
        pd.crosstab(case_level["gold_naics_code"], case_level["pred_naics_top1"], dropna=False)
        .sort_index()
        .sort_index(axis=1)
    )

    target_codes = ["327211", "327215"]
    pair_counts = {
        "327211->327211": int(((case_level["gold_naics_code"] == "327211") & (case_level["pred_naics_top1"] == "327211")).sum()),
        "327211->327215": int(((case_level["gold_naics_code"] == "327211") & (case_level["pred_naics_top1"] == "327215")).sum()),
        "327215->327215": int(((case_level["gold_naics_code"] == "327215") & (case_level["pred_naics_top1"] == "327215")).sum()),
        "327215->327211": int(((case_level["gold_naics_code"] == "327215") & (case_level["pred_naics_top1"] == "327211")).sum()),
    }
    predicted_distribution = case_level["pred_naics_top1"].value_counts(dropna=False).rename_axis("pred_naics_code").reset_index(name="count")
    predicted_distribution["share"] = predicted_distribution["count"] / max(len(case_level), 1)
    other_predictions = predicted_distribution.loc[~predicted_distribution["pred_naics_code"].isin(target_codes)].copy()

    hierarchy_wrong = case_level.loc[
        (~case_level["exact6_match"]) & (case_level["hier2_match"] | case_level["hier4_match"]),
        ["product_id", "gold_naics_code", "pred_naics_top1", "hier2_match", "hier4_match", "exact6_match", "stage_hint"],
    ].copy()
    topk_rescued = case_level.loc[
        (~case_level["exact6_match"]) & (case_level["topk_contains_gold"]),
        ["product_id", "gold_naics_code", "pred_naics_top1", "pred_naics_topk", "topk_contains_gold", "stage_hint"],
    ].copy()

    stage_slice = (
        case_level.groupby("stage_hint", dropna=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "case_count": int(len(frame)),
                    "gold_327211_count": int((frame["gold_naics_code"] == "327211").sum()),
                    "gold_327215_count": int((frame["gold_naics_code"] == "327215").sum()),
                    "pred_327211_count": int((frame["pred_naics_top1"] == "327211").sum()),
                    "pred_327215_count": int((frame["pred_naics_top1"] == "327215").sum()),
                    "327211_to_327215_count": int(
                        ((frame["gold_naics_code"] == "327211") & (frame["pred_naics_top1"] == "327215")).sum()
                    ),
                    "327215_to_327211_count": int(
                        ((frame["gold_naics_code"] == "327215") & (frame["pred_naics_top1"] == "327211")).sum()
                    ),
                    "exact6_error_count": int((~frame["exact6_match"]).sum()),
                    "topk_rescued_count": int(((~frame["exact6_match"]) & frame["topk_contains_gold"]).sum()),
                }
            )
        )
        .reset_index()
        .sort_values(["327211_to_327215_count", "327215_to_327211_count", "exact6_error_count", "stage_hint"], ascending=[False, False, False, True])
    )

    stage_to_327211 = stage_slice.loc[stage_slice["327215_to_327211_count"] > 0, ["stage_hint", "327215_to_327211_count"]]
    stage_to_327215 = stage_slice.loc[stage_slice["327211_to_327215_count"] > 0, ["stage_hint", "327211_to_327215_count"]]

    if stage_to_327211.empty:
        stage_to_327211_text = "none"
    else:
        stage_to_327211_text = ", ".join(
            f"{row.stage_hint} ({int(row['327215_to_327211_count'])})" for _, row in stage_to_327211.iterrows()
        )
    if stage_to_327215.empty:
        stage_to_327215_text = "none"
    else:
        stage_to_327215_text = ", ".join(
            f"{row.stage_hint} ({int(row['327211_to_327215_count'])})" for _, row in stage_to_327215.iterrows()
        )

    overall_exact_errors = int((~case_level["exact6_match"]).sum())
    rescued_count = int(((~case_level["exact6_match"]) & case_level["topk_contains_gold"]).sum())
    heatmap_path = out_dir / "02_confusion_heatmap.png"
    heatmap_written = _write_confusion_heatmap(confusion, heatmap_path)

    report_lines = [
        "# PV Glass Confusion Analysis",
        "",
        "## Input Check",
        "",
        f"- Source used: `{source_used}`.",
        f"- Case-level columns: `{list(case_level.columns)}`.",
        "- Gold label column: `gold_naics_code`.",
        "- Top-1 predicted label column: `pred_naics_top1`.",
        "",
        "## Core Confusion Counts",
        "",
        f"- `327211 -> 327211`: `{pair_counts['327211->327211']}`",
        f"- `327211 -> 327215`: `{pair_counts['327211->327215']}`",
        f"- `327215 -> 327215`: `{pair_counts['327215->327215']}`",
        f"- `327215 -> 327211`: `{pair_counts['327215->327211']}`",
        "",
        "## Other Predicted NAICS",
        "",
    ]
    if other_predictions.empty:
        report_lines.append("- No other NAICS codes were predicted outside `327211` and `327215`.")
    else:
        for _, row in other_predictions.iterrows():
            report_lines.append(
                f"- `{row['pred_naics_code']}`: `{int(row['count'])}` cases, share `{row['share']:.4f}`."
            )

    report_lines.extend(
        [
            "",
            "## Hierarchy-Aware Findings",
            "",
            f"- Cases with 6-digit error but still correct at 2-digit or 4-digit level: `{len(hierarchy_wrong)}`.",
            f"- Cases where top-1 was wrong but top-k still contained the gold NAICS: `{rescued_count}`.",
            "",
            "## Stage Slice",
            "",
            f"- Stages more likely to be mixed into `327211`: `{stage_to_327211_text}`.",
            f"- Stages more likely to be mixed into `327215`: `{stage_to_327215_text}`.",
            "",
            "## Conclusion",
            "",
        ]
    )

    if overall_exact_errors == 0:
        report_lines.extend(
            [
                "- The current automatic line does not show `327211 <-> 327215` confusion on this 36-case set. It is not mainly confusing raw flat-glass manufacturing with purchased-glass downstream processing in the current run.",
                "- No stage category stands out as harder because every `stage_hint` slice is classified correctly at 6 digits.",
                "- Top-k does not provide visible rescue on this run because top-1 is already exact on all cases.",
            ]
        )
    else:
        primary_confusion = "327211 <-> 327215" if pair_counts["327211->327215"] + pair_counts["327215->327211"] > 0 else "other NAICS categories"
        report_lines.extend(
            [
                f"- The main confusion pattern is `{primary_confusion}`.",
                f"- The hardest stage category is the one with the most exact-6 errors in [02_confusion_by_stage.csv]({out_dir / '02_confusion_by_stage.csv'}).",
                f"- Top-k rescues `{rescued_count}` of `{overall_exact_errors}` top-1 errors.",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The confusion matrix is built from top-1 NAICS predictions only.",
            "- Hierarchy-aware rescue uses the existing `hier2_match`, `hier4_match`, `exact6_match`, and `topk_contains_gold` fields from the case-level summary when available.",
            f"- Heatmap written: `{heatmap_written}`.",
        ]
    )

    ensure_directory(out_dir)
    confusion_path = out_dir / "02_confusion_matrix.csv"
    by_stage_path = out_dir / "02_confusion_by_stage.csv"
    report_path = out_dir / "02_confusion_report.md"

    confusion.to_csv(confusion_path, encoding="utf-8")
    stage_slice.to_csv(by_stage_path, index=False, encoding="utf-8")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    logger.info(
        "pv_glass_confusion_written",
        extra={
            "structured": {
                "confusion_path": str(confusion_path),
                "by_stage_path": str(by_stage_path),
                "report_path": str(report_path),
                "heatmap_written": heatmap_written,
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "case_count": int(len(case_level)),
            "exact6_errors": overall_exact_errors,
            "rescued_by_topk": rescued_count,
        },
    )


if __name__ == "__main__":
    main()
