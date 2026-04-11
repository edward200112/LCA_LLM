from __future__ import annotations

import math
from typing import Iterable


def _ranks_for_gold(candidates: list[dict], gold_code: str) -> list[int]:
    return [
        rank
        for rank, item in enumerate(candidates, start=1)
        if str(item.get("naics_code")) == str(gold_code)
    ]


def top1_accuracy(records: Iterable[dict]) -> float:
    records = list(records)
    hits = 0
    for record in records:
        candidates = record.get("candidates", [])
        if candidates and str(candidates[0].get("naics_code")) == str(record["gold_naics_code"]):
            hits += 1
    return hits / len(records) if records else 0.0


def recall_at_k(records: Iterable[dict], k: int) -> float:
    records = list(records)
    hits = 0
    for record in records:
        codes = [str(item.get("naics_code")) for item in record.get("candidates", [])[:k]]
        if str(record["gold_naics_code"]) in codes:
            hits += 1
    return hits / len(records) if records else 0.0


def mrr_at_k(records: Iterable[dict], k: int) -> float:
    records = list(records)
    total = 0.0
    for record in records:
        ranks = _ranks_for_gold(record.get("candidates", [])[:k], record["gold_naics_code"])
        total += 1.0 / ranks[0] if ranks else 0.0
    return total / len(records) if records else 0.0


def ndcg_at_k(records: Iterable[dict], k: int) -> float:
    records = list(records)
    total = 0.0
    for record in records:
        ranks = _ranks_for_gold(record.get("candidates", [])[:k], record["gold_naics_code"])
        total += 1.0 / math.log2(ranks[0] + 1) if ranks else 0.0
    return total / len(records) if records else 0.0


def hierarchical_accuracy(records: Iterable[dict], digits: int) -> float:
    records = list(records)
    hits = 0
    for record in records:
        candidates = record.get("candidates", [])
        if not candidates:
            continue
        pred = str(candidates[0].get("naics_code", ""))[:digits]
        gold = str(record["gold_naics_code"])[:digits]
        if pred == gold:
            hits += 1
    return hits / len(records) if records else 0.0


def compute_retrieval_metrics(records: list[dict]) -> dict[str, float]:
    return {
        "top1_accuracy": top1_accuracy(records),
        "recall@5": recall_at_k(records, 5),
        "recall@10": recall_at_k(records, 10),
        "mrr@10": mrr_at_k(records, 10),
        "ndcg@10": ndcg_at_k(records, 10),
        "hierarchical_accuracy@2": hierarchical_accuracy(records, 2),
        "hierarchical_accuracy@4": hierarchical_accuracy(records, 4),
        "hierarchical_accuracy@6": hierarchical_accuracy(records, 6),
    }
