from __future__ import annotations


def raise_lgbm_not_ready() -> None:
    raise RuntimeError(
        "LightGBM regression is scaffolded but not implemented in the current milestone."
    )
