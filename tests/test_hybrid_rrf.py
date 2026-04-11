from __future__ import annotations

from open_match_lca.retrieval.hybrid_rrf import reciprocal_rank_fusion


def test_rrf_prefers_consensus_candidates() -> None:
    fused = reciprocal_rank_fusion(
        [
            [
                {"candidate_id": "a", "naics_code": "111111", "score": 10.0},
                {"candidate_id": "b", "naics_code": "222222", "score": 9.0},
            ],
            [
                {"candidate_id": "b", "naics_code": "222222", "score": 8.0},
                {"candidate_id": "a", "naics_code": "111111", "score": 7.0},
            ],
        ],
        top_k=2,
    )
    assert fused[0]["candidate_id"] in {"a", "b"}
    assert len(fused) == 2
