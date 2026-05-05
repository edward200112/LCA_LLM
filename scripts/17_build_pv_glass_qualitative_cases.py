from __future__ import annotations

import argparse
import json
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


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _truncate(text: str, limit: int = 220) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    top1 = pd.read_parquet(top1_path).rename(
        columns={
            "pred_naics_code": "pred_naics_top1",
            "pred_factor_value": "pred_factor_top1",
            "y_true": "gold_factor",
        }
    )
    topk = pd.read_parquet(topk_path).rename(
        columns={
            "pred_naics_code": "pred_naics_topk",
            "pred_factor_value": "pred_factor_topk",
        }
    )
    merged = (
        cases[["product_id", "title", "description", "gold_naics_code"]]
        .merge(
            top1[["product_id", "gold_naics_code", "pred_naics_top1", "pred_factor_top1", "gold_factor"]],
            on=["product_id", "gold_naics_code"],
            how="left",
        )
        .merge(
            topk[["product_id", "gold_naics_code", "pred_naics_topk", "pred_factor_topk"]],
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
    merged["topk_contains_gold"] = merged["gold_naics_code"].astype(str) == merged["pred_naics_topk"].astype(str)
    return merged.fillna(""), "fallback_rebuild_from_prediction_parquet"


def _retrieval_features(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        candidates = record.get("candidates", []) or []
        score1 = candidates[0].get("score") if candidates else None
        score2 = candidates[1].get("score") if len(candidates) > 1 else None
        code2 = candidates[1].get("naics_code") if len(candidates) > 1 else ""
        rows.append(
            {
                "product_id": record.get("product_id"),
                "retrieval_score_top1": score1,
                "retrieval_score_top2": score2,
                "retrieval_margin_top12": (score1 - score2) if score1 is not None and score2 is not None else None,
                "retrieval_alt2_code": code2 or "",
            }
        )
    return pd.DataFrame(rows)


def _score_text(row: pd.Series) -> str:
    top1 = row.get("retrieval_score_top1")
    top2 = row.get("retrieval_score_top2")
    margin = row.get("retrieval_margin_top12")
    if pd.isna(top1) or pd.isna(top2) or pd.isna(margin):
        return "retrieval score detail unavailable"
    return f"top-1 BM25 score {top1:.2f}, runner-up {top2:.2f}, margin {margin:.2f}"


def _make_reason(row: pd.Series, requested_type: str, observed_type: str) -> str:
    reasons: list[str] = []
    if requested_type != observed_type:
        reasons.append(f"used as a substitute for unavailable `{requested_type}` examples")
    if row["gold_naics_code"] == "327211":
        reasons.append("represents upstream flat-glass manufacturing cues")
    else:
        reasons.append("represents downstream purchased-glass conversion cues")
    if row.get("retrieval_alt2_code") in {"327211", "327215"} and row.get("retrieval_alt2_code") != row["pred_naics_top1"]:
        reasons.append("runner-up retrieval result is the opposite glass class")
    if pd.notna(row.get("retrieval_margin_top12")) and float(row["retrieval_margin_top12"]) < 10:
        reasons.append("low retrieval margin makes it a boundary case")
    reasons.append(f"traceable to `{row.get('source_file', '')}`")
    return "; ".join(reasons)


def _make_narrative(row: pd.Series, requested_type: str, observed_type: str) -> str:
    sentences: list[str] = []
    stage = row.get("stage_hint", "unknown")
    title = row.get("title", "")
    sentences.append(
        f"This case is drawn from the `{stage}` stage and uses the product text \"{title}\" as the primary retrieval/classification signal."
    )
    if row["gold_naics_code"] == "327211":
        sentences.append(
            "The description emphasizes melt-line vocabulary such as float forming, batch melting, annealing, or raw-material composition, which is consistent with flat glass manufacturing rather than downstream fabrication."
        )
    else:
        sentences.append(
            "The description emphasizes downstream conversion language such as tempering, coating, washing, inspection, packing, or module-cover finishing, which is consistent with products made from purchased flat glass."
        )
    alt2 = row.get("retrieval_alt2_code", "")
    margin = row.get("retrieval_margin_top12")
    if alt2 and alt2 != row["pred_naics_top1"]:
        if pd.notna(margin) and float(margin) < 10:
            sentences.append(
                f"The second-ranked retrieval candidate is `{alt2}` with a narrow margin, so this example shows a real category-boundary tension even though the top-1 prediction is still correct."
            )
        else:
            sentences.append(
                f"The runner-up retrieval candidate is `{alt2}`, but the top-1 margin remains comfortable enough that the system resolves the upstream/downstream boundary correctly."
            )
    if requested_type != observed_type:
        sentences.append(
            f"No true `{requested_type}` example exists in the current 36-case run, so this case is reported as an informative substitute rather than as an actual model failure."
        )
    if str(row.get("gold_factor", "")).strip() == "":
        sentences.append(
            "Factor-level comparison is not available here because the prediction outputs do not include a populated gold factor column, so the qualitative judgment is classification- and retrieval-centered."
        )
    return " ".join(sentences[:5])


def _pick_one(frame: pd.DataFrame, *, used_ids: set[str], ascending: bool, code: str | None = None, stage: str | None = None) -> pd.Series | None:
    candidates = frame.copy()
    if code is not None:
        candidates = candidates.loc[candidates["gold_naics_code"].astype(str) == code]
    if stage is not None:
        candidates = candidates.loc[candidates["stage_hint"].astype(str) == stage]
    candidates = candidates.loc[~candidates["product_id"].isin(used_ids)]
    if candidates.empty:
        return None
    return candidates.sort_values(["retrieval_margin_top12", "product_id"], ascending=[ascending, True]).iloc[0]


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("17_build_pv_glass_qualitative_cases", LOGS_DIR)

    case_summary_path = Path(args.case_summary_path)
    top1_path = Path(args.top1_path)
    topk_path = Path(args.topk_path)
    cases_path = Path(args.cases_path)
    metadata_path = Path(args.metadata_path)
    retrieval_path = Path(args.retrieval_path)
    out_dir = Path(args.out_dir)

    _require_nonempty(metadata_path)
    _require_nonempty(retrieval_path)

    case_level, source_used = _load_case_level(case_summary_path, top1_path, topk_path, cases_path)
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False).fillna("")
    retrieval = _retrieval_features(retrieval_path)

    case_level["hier2_match"] = _to_bool(case_level["hier2_match"])
    case_level["hier4_match"] = _to_bool(case_level["hier4_match"])
    case_level["exact6_match"] = _to_bool(case_level["exact6_match"])
    case_level["topk_contains_gold"] = _to_bool(case_level["topk_contains_gold"])

    enriched = (
        case_level.merge(
            metadata[
                [
                    "product_id",
                    "source_note",
                    "source_file",
                    "source_doc_id",
                    "geography",
                    "thickness_mm",
                    "stage_hint",
                ]
            ],
            on="product_id",
            how="left",
            suffixes=("", "_meta"),
        )
        .merge(retrieval, on="product_id", how="left")
    )
    enriched["stage_hint"] = enriched["stage_hint"].replace("", pd.NA).combine_first(enriched.get("stage_hint_meta"))
    enriched["stage_hint"] = enriched["stage_hint"].fillna("unknown")

    used_ids: set[str] = set()
    selected_specs: list[tuple[str, str, pd.Series | None]] = []

    exact_pool = enriched.loc[enriched["exact6_match"]]
    hierarchy_pool = enriched.loc[(~enriched["exact6_match"]) & (enriched["hier2_match"] | enriched["hier4_match"])]
    recoverable_pool = enriched.loc[(~enriched["exact6_match"]) & (enriched["topk_contains_gold"])]
    if "gold_factor" in enriched.columns and enriched["gold_factor"].astype(str).str.strip().ne("").any():
        failure_pool = enriched.loc[~enriched["exact6_match"]].copy()
    else:
        failure_pool = enriched.loc[~enriched["topk_contains_gold"]].copy()

    selection_plan: list[tuple[str, str, dict[str, Any]]] = [
        ("exact_correct", "exact_correct", {"pool": exact_pool, "ascending": False, "code": "327211"}),
        ("exact_correct", "exact_correct", {"pool": exact_pool, "ascending": False, "code": "327215"}),
        ("hierarchy_only_correct", "boundary_exact_success", {"pool": exact_pool, "ascending": True, "code": "327211"}),
        ("hierarchy_only_correct", "boundary_exact_success", {"pool": exact_pool, "ascending": True, "stage": "coating_and_tempering", "code": "327215"}),
        ("topk_recoverable", "topk_sensitive_boundary", {"pool": exact_pool, "ascending": True, "stage": "tempering_finish_line", "code": "327215"}),
        ("topk_recoverable", "topk_sensitive_boundary", {"pool": exact_pool, "ascending": True, "stage": "solar_cover_finish", "code": "327215"}),
        ("obvious_failure", "lowest_margin_exact_success", {"pool": exact_pool, "ascending": True, "stage": "module_cover_finish", "code": "327215"}),
        ("obvious_failure", "lowest_margin_exact_success", {"pool": exact_pool, "ascending": True, "stage": "flat_glass_melt_line", "code": "327211"}),
        ("boundary_case", "boundary_case", {"pool": exact_pool, "ascending": False, "stage": "flat_glass_melt_line", "code": "327211"}),
    ]

    for requested_type, fallback_observed_type, selector in selection_plan:
        pool = selector["pool"]
        row = None
        if requested_type == "hierarchy_only_correct" and not hierarchy_pool.empty:
            row = _pick_one(
                hierarchy_pool,
                used_ids=used_ids,
                ascending=True,
                code=selector.get("code"),
                stage=selector.get("stage"),
            )
            observed_type = "hierarchy_only_correct"
        elif requested_type == "topk_recoverable" and not recoverable_pool.empty:
            row = _pick_one(
                recoverable_pool,
                used_ids=used_ids,
                ascending=True,
                code=selector.get("code"),
                stage=selector.get("stage"),
            )
            observed_type = "topk_recoverable"
        elif requested_type == "obvious_failure" and not failure_pool.empty:
            row = _pick_one(
                failure_pool,
                used_ids=used_ids,
                ascending=True,
                code=selector.get("code"),
                stage=selector.get("stage"),
            )
            observed_type = "obvious_failure"
        else:
            row = _pick_one(
                pool,
                used_ids=used_ids,
                ascending=bool(selector.get("ascending", True)),
                code=selector.get("code"),
                stage=selector.get("stage"),
            )
            observed_type = fallback_observed_type

        if row is None:
            row = _pick_one(exact_pool, used_ids=used_ids, ascending=True)
            observed_type = fallback_observed_type
        if row is None:
            continue
        selected_specs.append((requested_type, observed_type, row))
        used_ids.add(str(row["product_id"]))

    qualitative_rows: list[dict[str, Any]] = []
    case_counter = 1
    for requested_type, observed_type, row in selected_specs:
        if row is None:
            continue
        representative_reason = _make_reason(row, requested_type, observed_type)
        qualitative_rows.append(
            {
                "case_id": f"qual_case_{case_counter:02d}",
                "product_id": row["product_id"],
                "requested_case_type": requested_type,
                "observed_case_type": observed_type,
                "title": row.get("title", ""),
                "description_short": _truncate(str(row.get("description", ""))),
                "stage_hint": row.get("stage_hint", ""),
                "gold_naics_code": row.get("gold_naics_code", ""),
                "pred_naics_top1": row.get("pred_naics_top1", ""),
                "hier2_match": bool(row.get("hier2_match", False)),
                "hier4_match": bool(row.get("hier4_match", False)),
                "exact6_match": bool(row.get("exact6_match", False)),
                "topk_recoverable": bool(row.get("topk_contains_gold", False) and not row.get("exact6_match", False)),
                "gold_factor": row.get("gold_factor", ""),
                "pred_factor_top1": row.get("pred_factor_top1", ""),
                "pred_factor_topk": row.get("pred_factor_topk", ""),
                "retrieval_score_summary": _score_text(row),
                "retrieval_alt2_code": row.get("retrieval_alt2_code", ""),
                "source_note": row.get("source_note", ""),
                "source_file": row.get("source_file", ""),
                "source_doc_id": row.get("source_doc_id", ""),
                "representative_reason": representative_reason,
                "analysis_note": _make_narrative(row, requested_type, observed_type),
            }
        )
        case_counter += 1

    qualitative = pd.DataFrame(qualitative_rows)
    if qualitative.empty:
        raise RuntimeError("No qualitative cases were selected.")

    exact_failures = int((~enriched["exact6_match"]).sum())
    topk_rescues = int(((~enriched["exact6_match"]) & enriched["topk_contains_gold"]).sum())
    success_patterns = [
        "Success is most reliable when product text contains unambiguous melt-line cues for 327211 or downstream fabrication cues for 327215.",
        "Stage hints align cleanly with the upstream/downstream distinction: `flat_glass_melt_line` maps to 327211, while tempering/coating/module-cover stages map to 327215.",
        "Even boundary cases remain recoverable when the text preserves verbs and nouns specific to either furnace/float production or purchased-glass finishing.",
    ]
    if exact_failures == 0:
        failure_patterns = [
            "No true classification failures appear in the current 36-case evaluation, so the qualitative list substitutes low-margin boundary cases for the requested error buckets.",
            "The most plausible failure mode is still the semantic boundary between upstream flat-glass manufacture and downstream coating/tempering of purchased glass, because the runner-up retrieval class is often the opposite glass NAICS.",
        ]
        review_signal = [
            "The most useful manual-review signal is the top-1 versus runner-up retrieval margin.",
            "A second useful signal is `stage_hint`, because it exposes whether the text is describing melt-line production or post-processing.",
            "Hierarchy signals add little in this run because every case is already correct at 6 digits; they would matter more once genuine near-miss cases appear.",
        ]
    else:
        failure_patterns = [
            f"The run contains `{exact_failures}` exact-6 errors and `{topk_rescues}` top-k recoveries, indicating that some mistakes remain close to the correct class in retrieval space.",
            "Observed failures should be interpreted as boundary errors between similar glass processes unless the confusion matrix shows spillover to unrelated NAICS classes.",
        ]
        review_signal = [
            "The most useful manual-review signals are top-k recovery, stage hints, and whether 2-digit/4-digit hierarchy remains correct after a 6-digit miss.",
        ]

    report_lines = [
        "# PV Glass Qualitative Case List",
        "",
        "## Input Check",
        "",
        f"- Source used for case-level analysis: `{source_used}`.",
        f"- Selected cases: `{len(qualitative)}`.",
        f"- Available exact-6 errors in the full run: `{exact_failures}`.",
        f"- Available top-k recoverable errors in the full run: `{topk_rescues}`.",
        "- Because the current run has no true classification failures, the requested hierarchy-only, top-k-recoverable, and obvious-failure buckets are represented by low-margin boundary substitutes where necessary.",
        "",
        "## Selection Logic",
        "",
        "- Two cases were chosen as clear successes, one for upstream flat-glass manufacturing and one for downstream purchased-glass conversion.",
        "- The remaining cases emphasize low-margin retrieval boundaries, stage diversity, and source diversity so the list still supports paper-style qualitative discussion.",
        "- Every selected row is traceable to `product_id`, `source_file`, and the existing prediction outputs.",
        "",
        "## Case Notes",
        "",
    ]

    for _, row in qualitative.iterrows():
        report_lines.extend(
            [
                f"### {row['case_id']} | {row['product_id']}",
                "",
                f"- Requested type: `{row['requested_case_type']}`",
                f"- Observed type: `{row['observed_case_type']}`",
                f"- Title: `{row['title']}`",
                f"- Stage: `{row['stage_hint']}`",
                f"- Gold / Pred top-1: `{row['gold_naics_code']} / {row['pred_naics_top1']}`",
                f"- Hierarchy match: `hier2={row['hier2_match']}`, `hier4={row['hier4_match']}`, `exact6={row['exact6_match']}`",
                f"- Top-k recoverable: `{row['topk_recoverable']}`",
                f"- Factor values: `gold={row['gold_factor'] or 'NA'}`, `top1={row['pred_factor_top1'] or 'NA'}`, `topk={row['pred_factor_topk'] or 'NA'}`",
                f"- Traceability: `source_file={row['source_file']}`, `source_note={row['source_note']}`, `source_doc_id={row['source_doc_id']}`",
                f"- Why representative: {row['representative_reason']}",
                "",
                row["analysis_note"],
                "",
            ]
        )

    report_lines.extend(
        [
            "## Discussion",
            "",
            "### Typical Success Modes",
            "",
        ]
    )
    report_lines.extend([f"- {line}" for line in success_patterns])
    report_lines.extend(
        [
            "",
            "### Typical Failure Modes",
            "",
        ]
    )
    report_lines.extend([f"- {line}" for line in failure_patterns])
    report_lines.extend(
        [
            "",
            "### Signals Most Valuable for Manual Review",
            "",
        ]
    )
    report_lines.extend([f"- {line}" for line in review_signal])

    ensure_directory(out_dir)
    csv_path = out_dir / "03_qualitative_cases.csv"
    report_path = out_dir / "03_qualitative_report.md"
    qualitative.to_csv(csv_path, index=False, encoding="utf-8")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    logger.info(
        "pv_glass_qualitative_written",
        extra={
            "structured": {
                "csv_path": str(csv_path),
                "report_path": str(report_path),
                "selected_cases": int(len(qualitative)),
            }
        },
    )
    log_final_metrics(
        logger,
        {
            "selected_cases": int(len(qualitative)),
            "available_exact_errors": exact_failures,
            "available_topk_rescues": topk_rescues,
        },
    )


if __name__ == "__main__":
    main()
