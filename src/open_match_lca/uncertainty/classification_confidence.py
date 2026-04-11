from __future__ import annotations

import numpy as np

from open_match_lca.regression.topk_factor_mixture import softmax


def confidence_from_scores(scores: list[float]) -> dict[str, float]:
    probs = softmax(scores)
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "top1_probability": top1,
        "top1_top2_margin": top1 - top2,
        "entropy": entropy,
    }
