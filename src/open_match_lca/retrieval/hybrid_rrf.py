from __future__ import annotations

from collections import defaultdict

from open_match_lca.constants import RRF_K


def reciprocal_rank_fusion(
    runs: list[list[dict[str, float | str]]],
    top_k: int = 50,
    k: int = RRF_K,
) -> list[dict[str, float | str]]:
    scores: dict[str, float] = defaultdict(float)
    first_payload: dict[str, dict[str, float | str]] = {}
    for run in runs:
        for rank, item in enumerate(run, start=1):
            candidate_id = str(item["candidate_id"])
            scores[candidate_id] += 1.0 / (k + rank)
            first_payload.setdefault(candidate_id, item)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            **first_payload[candidate_id],
            "rrf_score": score,
        }
        for candidate_id, score in ranked
    ]
