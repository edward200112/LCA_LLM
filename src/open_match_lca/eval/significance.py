from __future__ import annotations

import numpy as np


def mean_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Expected equal length inputs, got {len(left)} and {len(right)}")
    return float(np.mean(np.asarray(left) - np.asarray(right)))
