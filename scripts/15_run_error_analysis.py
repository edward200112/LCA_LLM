from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import dump_json, ensure_directory, read_jsonl, write_jsonl, write_parquet
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger

BRAND_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "inch",
    "pack",
    "size",
    "set",
    "kit",
    "black",
    "white",
    "blue",
    "red",
}
SPEC_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s?(oz|inch|in|cm|mm|ml|l|lb|lbs|w|watt|v|volt|mah|gb|tb|pack|count|ct|pcs|pc|ft|xl|xxl|xlarge)\b",
    flags=re.IGNORECASE,
)
DIGIT_PATTERN = re.compile(r"\d")
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products_path", required=True)
    parser.add_argument("--train_path", required=False)
    parser.add_argument("--retrieval_path", required=True)
    parser.add_argument("--baseline_retrieval_path", required=False)
    parser.add_argument("--regressor_pred_path", required=False)
    parser.add_argument("--top1_pred_path", required=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--case_limit", type=int, default=200)
    return parser


def _token_count(text: str) -> int:
    return len(text.split())


def _has_brand_like_token(text: str) -> bool:
    for token in TOKEN_PATTERN.findall(text):
        if token.lower() in BRAND_STOPWORDS:
            continue
        if any(char.isdigit() for char in token):
            continue
        if token.isupper() or any(char.isupper() for char in token[1:]):
            return True
    return False


def _has_spec_terms(text: str) -> bool:
    return bool(SPEC_PATTERN.search(text))


def _frequency_slice(count: int, low_cutoff: float, high_cutoff: float) -> str:
    if count <= low_cutoff:
        return "tail"
    if count >= high_cutoff:
        return "head"
    return "mid"


def _prepare_product_features(products: pd.DataFrame, train_frame: pd.DataFrame | None) -> pd.DataFrame:
    frame = products.copy()
    frame["product_id"] = frame["product_id"].astype(str)
    frame["gold_naics_code"] = frame["gold_naics_code"].astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["text_len_tokens"] = frame["text"].map(_token_count)
    text_median = float(frame["text_len_tokens"].median()) if not frame.empty else 0.0
    frame["text_length_slice"] = np.where(frame["text_len_tokens"] <= text_median, "short", "long")
    frame["has_digit_heuristic"] = frame["text"].str.contains(DIGIT_PATTERN)
    frame["has_spec_terms"] = frame["text"].map(_has_spec_terms)
    frame["has_brand_like_token"] = frame["text"].map(_has_brand_like_token)

    if train_frame is None:
        counts = frame["gold_naics_code"].value_counts()
    else:
        counts = train_frame["gold_naics_code"].astype(str).value_counts()
    low_cutoff = float(counts.quantile(0.25)) if not counts.empty else 0.0
    high_cutoff = float(counts.quantile(0.75)) if not counts.empty else 0.0
    frame["train_naics_frequency"] = frame["gold_naics_code"].map(counts).fillna(0).astype(int)
    frame["naics_frequency_slice"] = frame["train_naics_frequency"].map(
        lambda value: _frequency_slice(int(value), low_cutoff, high_cutoff)
    )
    return frame


def _top_candidate(record: dict) -> dict | None:
    candidates = record.get("candidates", [])
    return candidates[0] if candidates else None


def _retrieval_frame(records: list[dict], products: pd.DataFrame, top_k: int) -> pd.DataFrame:
    product_meta = products.set_index("product_id").to_dict("index")
    rows: list[dict[str, object]] = []
    for record in records:
        product_id = str(record["product_id"])
        meta = product_meta[product_id]
        candidates = record.get("candidates", [])[:top_k]
        top = _top_candidate(record)
        pred_code = None if top is None else str(top.get("naics_code"))
        gold_code = str(record.get("gold_naics_code", meta["gold_naics_code"]))
        rows.append(
            {
                "product_id": product_id,
                "gold_naics_code": gold_code,
                "pred_naics_code": pred_code,
                "top1_correct": int(pred_code == gold_code),
                "topk_contains_truth": int(gold_code in {str(item.get("naics_code")) for item in candidates}),
                "hier2_correct": int((pred_code or "")[:2] == gold_code[:2]),
                "hier4_correct": int((pred_code or "")[:4] == gold_code[:4]),
                "text": meta["text"],
                "text_len_tokens": meta["text_len_tokens"],
                "text_length_slice": meta["text_length_slice"],
                "has_digit_heuristic": int(meta["has_digit_heuristic"]),
                "has_spec_terms": int(meta["has_spec_terms"]),
                "has_brand_like_token": int(meta["has_brand_like_token"]),
                "naics_frequency_slice": meta["naics_frequency_slice"],
                "train_naics_frequency": meta["train_naics_frequency"],
                "candidate_count": len(candidates),
            }
        )
    return pd.DataFrame(rows)


def _slice_summary(frame: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    slice_specs = {
        "text_length_slice": ["short", "long"],
        "has_brand_like_token": [0, 1],
        "has_spec_terms": [0, 1],
        "has_digit_heuristic": [0, 1],
        "naics_frequency_slice": ["head", "tail"],
    }
    rows: list[dict[str, object]] = []
    for column, values in slice_specs.items():
        for value in values:
            subset = frame.loc[frame[column] == value].copy()
            if subset.empty:
                continue
            row: dict[str, object] = {
                "slice_name": column,
                "slice_value": value,
                "n": int(len(subset)),
            }
            for metric in metric_cols:
                values_series = pd.to_numeric(subset[metric], errors="coerce").dropna()
                if values_series.empty:
                    continue
                row[metric] = float(values_series.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _export_case_records(frame: pd.DataFrame, path: Path, limit: int) -> int:
    if frame.empty:
        write_jsonl([], path)
        return 0
    export_frame = frame.head(limit).copy()
    write_jsonl(export_frame.to_dict(orient="records"), path)
    return int(len(export_frame))


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("15_run_error_analysis", LOGS_DIR)
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)

    products = pd.read_parquet(args.products_path)
    train_frame = pd.read_parquet(args.train_path) if args.train_path else None
    product_features = _prepare_product_features(products, train_frame)

    retrieval_records = read_jsonl(args.retrieval_path)
    retrieval_frame = _retrieval_frame(retrieval_records, product_features, args.top_k)
    retrieval_slice_summary = _slice_summary(
        retrieval_frame,
        ["top1_correct", "topk_contains_truth", "hier2_correct", "hier4_correct"],
    )
    retrieval_slice_summary.to_csv(output_dir / "retrieval_slice_summary.csv", index=False)
    dump_json({"rows": retrieval_slice_summary.to_dict(orient="records")}, output_dir / "retrieval_slice_summary.json")

    topk_recoverable = retrieval_frame.loc[
        (retrieval_frame["top1_correct"] == 0) & (retrieval_frame["topk_contains_truth"] == 1)
    ].sort_values(["text_len_tokens", "train_naics_frequency"], ascending=[False, True])
    hierarchy_partial = retrieval_frame.loc[
        (retrieval_frame["hier2_correct"] == 1) & (retrieval_frame["top1_correct"] == 0)
    ].sort_values(["text_len_tokens", "train_naics_frequency"], ascending=[False, True])

    exported_cases = {
        "topk_recoverable_cases": _export_case_records(
            topk_recoverable,
            output_dir / "cases_top1_wrong_but_topk_contains_truth.jsonl",
            args.case_limit,
        ),
        "hier2_correct_6_wrong_cases": _export_case_records(
            hierarchy_partial,
            output_dir / "cases_hier2_correct_but_6digit_wrong.jsonl",
            args.case_limit,
        ),
    }

    if args.baseline_retrieval_path:
        baseline_records = read_jsonl(args.baseline_retrieval_path)
        baseline_frame = _retrieval_frame(baseline_records, product_features, args.top_k)
        merged = baseline_frame.merge(
            retrieval_frame,
            on="product_id",
            suffixes=("_before", "_after"),
            how="inner",
        )
        reranker_fixed = merged.loc[
            (merged["top1_correct_before"] == 0)
            & (merged["top1_correct_after"] == 1)
        ].copy()
        reranker_fixed = reranker_fixed.rename(
            columns={
                "gold_naics_code_after": "gold_naics_code",
                "pred_naics_code_before": "pred_naics_code_before",
                "pred_naics_code_after": "pred_naics_code_after",
                "text_after": "text",
            }
        )
        exported_cases["reranker_fixed_cases"] = _export_case_records(
            reranker_fixed.loc[
                :,
                [
                    "product_id",
                    "gold_naics_code",
                    "pred_naics_code_before",
                    "pred_naics_code_after",
                    "text",
                    "text_len_tokens_after",
                    "has_digit_heuristic_after",
                    "has_spec_terms_after",
                    "has_brand_like_token_after",
                    "naics_frequency_slice_after",
                ],
            ],
            output_dir / "cases_reranker_fixed.jsonl",
            args.case_limit,
        )

    regression_slice_summary = pd.DataFrame()
    if args.regressor_pred_path:
        regressor = pd.read_parquet(args.regressor_pred_path).copy()
        regressor["product_id"] = regressor["product_id"].astype(str)
        regressor = regressor.merge(
            product_features[
                [
                    "product_id",
                    "text",
                    "text_len_tokens",
                    "text_length_slice",
                    "has_digit_heuristic",
                    "has_spec_terms",
                    "has_brand_like_token",
                    "naics_frequency_slice",
                    "train_naics_frequency",
                ]
            ],
            on="product_id",
            how="left",
        )
        if "error" not in regressor.columns and {"y_true", "pred_factor_value"}.issubset(regressor.columns):
            regressor["error"] = (regressor["pred_factor_value"] - regressor["y_true"]).abs()
        regression_slice_summary = _slice_summary(
            regressor,
            ["error", "confidence", "retained"],
        )
        regression_slice_summary = regression_slice_summary.rename(columns={"error": "mae"})
        regression_slice_summary.to_csv(output_dir / "regression_slice_summary.csv", index=False)
        dump_json({"rows": regression_slice_summary.to_dict(orient="records")}, output_dir / "regression_slice_summary.json")

        abstained_high_risk = regressor.loc[~regressor["retained"].astype(bool)].copy() if "retained" in regressor.columns else pd.DataFrame()
        if not abstained_high_risk.empty and "error" in abstained_high_risk.columns:
            threshold = float(abstained_high_risk["error"].quantile(0.75))
            abstained_high_risk = abstained_high_risk.loc[abstained_high_risk["error"] >= threshold].sort_values(
                "error",
                ascending=False,
            )
            write_parquet(
                abstained_high_risk.head(args.case_limit),
                output_dir / "cases_abstained_high_risk.parquet",
            )
            exported_cases["abstained_high_risk_cases"] = int(min(len(abstained_high_risk), args.case_limit))

        if args.top1_pred_path:
            top1 = pd.read_parquet(args.top1_pred_path).copy()
            top1["product_id"] = top1["product_id"].astype(str)
            if "error" not in top1.columns and {"y_true", "pred_factor_value"}.issubset(top1.columns):
                top1["error"] = (top1["pred_factor_value"] - top1["y_true"]).abs()
            improved = regressor.merge(
                top1.loc[:, ["product_id", "pred_factor_value", "error"]],
                on="product_id",
                how="inner",
                suffixes=("_regressor", "_top1"),
            )
            improved["error_gain"] = improved["error_top1"] - improved["error_regressor"]
            improved = improved.loc[improved["error_gain"] > 0].sort_values("error_gain", ascending=False)
            write_parquet(improved.head(args.case_limit), output_dir / "cases_regressor_better_than_top1.parquet")
            exported_cases["regressor_improvement_cases"] = int(min(len(improved), args.case_limit))

    summary = {
        "retrieval_rows": int(len(retrieval_frame)),
        "retrieval_slice_rows": int(len(retrieval_slice_summary)),
        "regression_slice_rows": int(len(regression_slice_summary)),
        "exported_cases": exported_cases,
    }
    dump_json(summary, output_dir / "error_analysis_summary.json")
    log_final_metrics(logger, summary)


if __name__ == "__main__":
    main()
