from __future__ import annotations


def raise_reranker_not_ready() -> None:
    raise RuntimeError(
        "Cross-encoder reranking is scaffolded but not yet implemented in this milestone."
    )
