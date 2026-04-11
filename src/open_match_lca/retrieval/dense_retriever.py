from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from open_match_lca.io_utils import ensure_directory

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None


class EncoderLike(Protocol):
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        ...


@dataclass
class DenseSearchHit:
    index: int
    score: float


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class DenseRetriever:
    def __init__(
        self,
        corpus_texts: list[str],
        encoder_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        encoder: EncoderLike | None = None,
    ) -> None:
        self.corpus_texts = corpus_texts
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.encoder = encoder or self._load_encoder(encoder_name)
        self.corpus_embeddings = self._encode(corpus_texts)

    def _load_encoder(self, encoder_name: str) -> EncoderLike:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install project full dependencies "
                "before using dense retrieval."
            ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR
        return SentenceTransformer(encoder_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Expected 2D embeddings from encoder {self.encoder_name}, got shape {embeddings.shape}"
            )
        return _normalize_rows(embeddings)

    def search(self, query: str, top_k: int = 50) -> list[DenseSearchHit]:
        query_embedding = self._encode([query])[0]
        scores = self.corpus_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            DenseSearchHit(index=int(index), score=float(scores[index]))
            for index in top_indices
        ]

    def search_batch(self, queries: list[str], top_k: int = 50) -> list[list[DenseSearchHit]]:
        query_embeddings = self._encode(queries)
        score_matrix = query_embeddings @ self.corpus_embeddings.T
        results: list[list[DenseSearchHit]] = []
        for row_scores in score_matrix:
            top_indices = np.argsort(row_scores)[::-1][:top_k]
            results.append(
                [
                    DenseSearchHit(index=int(index), score=float(row_scores[index]))
                    for index in top_indices
                ]
            )
        return results

    def save_index(self, index_dir: str | Path) -> tuple[Path, Path]:
        index_root = Path(index_dir)
        ensure_directory(index_root)
        embeddings_path = index_root / "dense_embeddings.npy"
        texts_path = index_root / "dense_corpus_texts.txt"
        np.save(embeddings_path, self.corpus_embeddings)
        texts_path.write_text("\n".join(self.corpus_texts), encoding="utf-8")
        return embeddings_path, texts_path
