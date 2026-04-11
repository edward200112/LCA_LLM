from __future__ import annotations


def raise_predict_regression_not_ready() -> None:
    raise RuntimeError(
        "Regression prediction from retrieval artifacts is scaffolded but not implemented yet."
    )
