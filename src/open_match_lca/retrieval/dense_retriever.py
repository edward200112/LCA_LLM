from __future__ import annotations


def raise_dense_not_ready() -> None:
    raise RuntimeError(
        "Dense retrieval training/inference is reserved for the next implementation milestone. "
        "This scaffold intentionally ships without a default online or closed-source dependency."
    )
