from __future__ import annotations


def raise_xgb_not_ready() -> None:
    raise RuntimeError(
        "XGBoost regression is scaffolded but not implemented in the current milestone."
    )
