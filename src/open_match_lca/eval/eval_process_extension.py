from __future__ import annotations

import pandas as pd

from open_match_lca.eval.eval_retrieval import mrr_at_k, recall_at_k


def evaluate_process_extension(records: list[dict], has_silver_labels: bool) -> dict | pd.DataFrame:
    if has_silver_labels:
        return {
            "recall@5": recall_at_k(records, 5),
            "recall@10": recall_at_k(records, 10),
            "mrr@10": mrr_at_k(records, 10),
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
