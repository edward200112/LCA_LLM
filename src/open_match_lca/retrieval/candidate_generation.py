from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from open_match_lca.features.text_cleaning import clean_text
from open_match_lca.retrieval.bm25_retriever import BM25Retriever
from open_match_lca.retrieval.dense_retriever import DenseRetriever, EncoderLike


def lexical_overlap_score(query: str, candidate: str) -> float:
    query_tokens = set(clean_text(query).lower().split())
    candidate_tokens = set(clean_text(candidate).lower().split())
    if not query_tokens or not candidate_tokens:
        return 0.0
    return float(len(query_tokens & candidate_tokens) / len(query_tokens))


def exact_or_lexical_retrieve(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    top_k: int,
) -> list[dict]:
    outputs: list[dict] = []
    for row in queries.itertuples(index=False):
        candidates = []
        for candidate in corpus.itertuples(index=False):
            score = lexical_overlap_score(row.text, candidate.naics_text)
            candidates.append(
                {
                    "candidate_id": candidate.naics_code,
                    "naics_code": candidate.naics_code,
                    "score": score,
                }
            )
        candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]
        outputs.append(
            {
                "product_id": row.product_id,
                "gold_naics_code": row.gold_naics_code,
                "query_text": row.text,
                "candidates": candidates,
            }
        )
    return outputs


def tfidf_retrieve(queries: pd.DataFrame, corpus: pd.DataFrame, top_k: int) -> list[dict]:
    vectorizer = TfidfVectorizer(min_df=1)
    corpus_matrix = vectorizer.fit_transform(corpus["naics_text"].astype(str))
    query_matrix = vectorizer.transform(queries["text"].astype(str))
    scores = linear_kernel(query_matrix, corpus_matrix)
    outputs: list[dict] = []
    for row_index, row in enumerate(queries.itertuples(index=False)):
        row_scores = np.asarray(scores[row_index], dtype=float)
        top_indices = row_scores.argsort()[::-1][:top_k]
        candidates = [
            {
                "candidate_id": corpus.iloc[index]["naics_code"],
                "naics_code": corpus.iloc[index]["naics_code"],
                "score": float(row_scores[index]),
            }
            for index in top_indices
        ]
        outputs.append(
            {
                "product_id": row.product_id,
                "gold_naics_code": row.gold_naics_code,
                "query_text": row.text,
                "candidates": candidates,
            }
        )
    return outputs


def bm25_retrieve(queries: pd.DataFrame, corpus: pd.DataFrame, top_k: int) -> list[dict]:
    retriever = BM25Retriever(corpus["naics_text"].astype(str).tolist())
    outputs: list[dict] = []
    for row in queries.itertuples(index=False):
        hits = retriever.search(row.text, top_k=top_k)
        candidates = [
            {
                "candidate_id": corpus.iloc[index]["naics_code"],
                "naics_code": corpus.iloc[index]["naics_code"],
                "score": float(score),
            }
            for index, score in hits
        ]
        outputs.append(
            {
                "product_id": row.product_id,
                "gold_naics_code": row.gold_naics_code,
                "query_text": row.text,
                "candidates": candidates,
            }
        )
    return outputs


def dense_zero_shot_retrieve(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    top_k: int,
    encoder_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    encoder: EncoderLike | None = None,
    index_dir: str | None = None,
    device: str | None = None,
    show_progress_bar: bool = True,
) -> list[dict]:
    retriever = DenseRetriever(
        corpus_texts=corpus["naics_text"].astype(str).tolist(),
        encoder_name=encoder_name,
        batch_size=batch_size,
        encoder=encoder,
        device=device,
        show_progress_bar=show_progress_bar,
    )
    if index_dir:
        retriever.save_index(index_dir)
    batch_hits = retriever.search_batch(queries["text"].astype(str).tolist(), top_k=top_k)
    outputs: list[dict] = []
    for row, hits in zip(queries.itertuples(index=False), batch_hits, strict=False):
        candidates = [
            {
                "candidate_id": corpus.iloc[hit.index]["naics_code"],
                "naics_code": corpus.iloc[hit.index]["naics_code"],
                "score": float(hit.score),
            }
            for hit in hits
        ]
        outputs.append(
            {
                "product_id": row.product_id,
                "gold_naics_code": row.gold_naics_code,
                "query_text": row.text,
                "candidates": candidates,
            }
        )
    return outputs
