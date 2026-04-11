from __future__ import annotations

import numpy as np
import pandas as pd

from open_match_lca.retrieval.rerank_cross_encoder import (
    build_reranker_pairs_from_run,
    rerank_retrieval_records,
    scored_pairs_to_retrieval_records,
)


class FakeScorer:
    def predict(
        self,
        inputs: list[list[str]],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        scores = []
        for query_text, candidate_text in inputs:
            overlap = len(set(query_text.lower().split()) & set(candidate_text.lower().split()))
            scores.append(float(overlap))
        return np.asarray(scores, dtype=float)


def test_build_reranker_pairs_from_run() -> None:
    records = [
        {
            "product_id": "p1",
            "gold_naics_code": "337127",
            "query_text": "metal chair",
            "candidates": [
                {"candidate_id": "337127", "naics_code": "337127", "score": 1.0},
                {"candidate_id": "335139", "naics_code": "335139", "score": 0.5},
            ],
        }
    ]
    corpus = pd.DataFrame(
        [
            {"naics_code": "337127", "naics_text": "metal chair furniture"},
            {"naics_code": "335139", "naics_text": "electric lamp products"},
        ]
    )
    pairs = build_reranker_pairs_from_run(records, corpus, top_k=2)
    assert len(pairs) == 2
    assert pairs["label"].tolist() == [1.0, 0.0]
    assert "candidate_text" in pairs.columns


def test_rerank_retrieval_records_with_fake_scorer() -> None:
    records = [
        {
            "product_id": "p1",
            "gold_naics_code": "337127",
            "query_text": "metal chair",
            "candidates": [
                {"candidate_id": "335139", "naics_code": "335139", "score": 10.0},
                {"candidate_id": "337127", "naics_code": "337127", "score": 9.0},
            ],
        }
    ]
    corpus = pd.DataFrame(
        [
            {"naics_code": "337127", "naics_text": "metal chair furniture"},
            {"naics_code": "335139", "naics_text": "electric lamp products"},
        ]
    )
    reranked = rerank_retrieval_records(
        retrieval_records=records,
        corpus=corpus,
        model_name_or_path="fake",
        scorer=FakeScorer(),
        top_k=2,
    )
    assert reranked[0]["candidates"][0]["naics_code"] == "337127"
    assert "rerank_score" in reranked[0]["candidates"][0]


def test_scored_pairs_to_retrieval_records_groups_and_sorts() -> None:
    scored_pairs = pd.DataFrame(
        [
            {
                "product_id": "p1",
                "gold_naics_code": "337127",
                "query_text": "metal chair",
                "candidate_id": "335139",
                "naics_code": "335139",
                "initial_score": 0.1,
                "rerank_score": 0.2,
            },
            {
                "product_id": "p1",
                "gold_naics_code": "337127",
                "query_text": "metal chair",
                "candidate_id": "337127",
                "naics_code": "337127",
                "initial_score": 0.05,
                "rerank_score": 0.9,
            },
        ]
    )
    records = scored_pairs_to_retrieval_records(scored_pairs, top_k=2)
    assert len(records) == 1
    assert records[0]["candidates"][0]["naics_code"] == "337127"
