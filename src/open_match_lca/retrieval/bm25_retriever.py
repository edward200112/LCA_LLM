from __future__ import annotations

import numpy as np

from open_match_lca.features.text_cleaning import clean_text

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    BM25Okapi = None
    _BM25_IMPORT_ERROR = exc
else:
    _BM25_IMPORT_ERROR = None


class BM25Retriever:
    def __init__(self, corpus_texts: list[str]) -> None:
        if BM25Okapi is None:
            raise RuntimeError(
                "rank-bm25 is not installed. Install project dependencies before using BM25."
            ) from _BM25_IMPORT_ERROR
        self.corpus_texts = corpus_texts
        self.tokenized_corpus = [clean_text(text).lower().split() for text in corpus_texts]
        self.model = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        scores = np.asarray(self.model.get_scores(clean_text(query).lower().split()), dtype=float)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(index), float(scores[index])) for index in top_indices]
