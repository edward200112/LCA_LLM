from __future__ import annotations

import numpy as np
import pandas as pd

from open_match_lca.retrieval.candidate_generation import dense_zero_shot_retrieve


class FakeEncoder:
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        vocab = ["chair", "desk", "lamp", "metal", "wood"]
        matrix = []
        for sentence in sentences:
            lowered = sentence.lower()
            matrix.append([float(token in lowered) for token in vocab])
        return np.asarray(matrix, dtype=float)


def test_dense_zero_shot_retrieve_with_fake_encoder() -> None:
    queries = pd.DataFrame(
        [
            {"product_id": "p1", "text": "metal chair", "gold_naics_code": "111111"},
            {"product_id": "p2", "text": "wood desk", "gold_naics_code": "222222"},
        ]
    )
    corpus = pd.DataFrame(
        [
            {"naics_code": "111111", "naics_text": "metal chair furniture"},
            {"naics_code": "222222", "naics_text": "wood desk furniture"},
            {"naics_code": "333333", "naics_text": "electric lamp products"},
        ]
    )
    runs = dense_zero_shot_retrieve(
        queries,
        corpus,
        top_k=2,
        encoder_name="fake",
        batch_size=2,
        encoder=FakeEncoder(),
    )
    assert runs[0]["candidates"][0]["naics_code"] == "111111"
    assert runs[1]["candidates"][0]["naics_code"] == "222222"
