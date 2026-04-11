from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def reduce_embeddings_pca(embeddings: np.ndarray, n_components: int = 64) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")
    actual_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    return PCA(n_components=actual_components, random_state=0).fit_transform(embeddings)
