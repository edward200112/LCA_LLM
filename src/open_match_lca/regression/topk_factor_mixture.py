from __future__ import annotations

import math

import numpy as np


def softmax(scores: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=float)
    shifted = array - np.max(array)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if math.isclose(float(denom), 0.0):
        raise ValueError(f"Cannot compute softmax for scores: {scores}")
    return exps / denom


def topk_factor_mixture(factor_values: list[float], scores: list[float]) -> dict[str, float]:
    if len(factor_values) != len(scores):
        raise ValueError(
            f"factor_values and scores must have the same length, got "
            f"{len(factor_values)} and {len(scores)}"
        )
    probs = softmax(scores)
    factors = np.asarray(factor_values, dtype=float)
    weighted = probs * factors
    return {
        "prediction": float(weighted.sum()),
        "prob_max": float(probs.max()),
        "factor_mean": float(factors.mean()),
        "factor_std": float(factors.std()),
        "factor_min": float(factors.min()),
        "factor_max": float(factors.max()),
    }
