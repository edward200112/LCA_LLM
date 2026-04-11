from __future__ import annotations

import pandas as pd

from open_match_lca.io_utils import dump_json


def _process_recall_at_k(records: list[dict], k: int) -> float:
    if not records:
        return 0.0
    hits = 0
    for record in records:
        gold_uuid = str(record.get("gold_process_uuid", ""))
        candidates = [str(item.get("process_uuid", "")) for item in record.get("candidates", [])[:k]]
        if gold_uuid and gold_uuid in candidates:
            hits += 1
    return hits / len(records)


def _process_mrr_at_k(records: list[dict], k: int) -> float:
    if not records:
        return 0.0
    total = 0.0
    for record in records:
        gold_uuid = str(record.get("gold_process_uuid", ""))
        reciprocal = 0.0
        for rank, candidate in enumerate(record.get("candidates", [])[:k], start=1):
            if str(candidate.get("process_uuid", "")) == gold_uuid and gold_uuid:
                reciprocal = 1.0 / rank
                break
        total += reciprocal
    return total / len(records)


def evaluate_process_extension(records: list[dict], has_silver_labels: bool) -> dict | pd.DataFrame:
    if has_silver_labels:
        return {
            "recall@5": _process_recall_at_k(records, 5),
            "recall@10": _process_recall_at_k(records, 10),
            "mrr@10": _process_mrr_at_k(records, 10),
        }
    rows = []
    for record in records:
        rows.append(
            {
                "product_id": record["product_id"],
                "query_text": record.get("query_text", ""),
                "topk_process_names": [item.get("process_name") for item in record.get("candidates", [])],
                "scores": [item.get("score") for item in record.get("candidates", [])],
                "geography": [item.get("geography") for item in record.get("candidates", [])],
                "reference_flow_name": [item.get("reference_flow_name") for item in record.get("candidates", [])],
            }
        )
    return pd.DataFrame(rows)


def export_process_extension_outputs(
    records: list[dict],
    has_silver_labels: bool,
    output_path: str,
) -> None:
    output = evaluate_process_extension(records, has_silver_labels)
    if isinstance(output, dict):
        dump_json(output, output_path)
        return
    output.to_parquet(output_path, index=False)
