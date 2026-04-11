from __future__ import annotations

from pathlib import Path


def build_model_card(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    output_path.write_text(content, encoding="utf-8")
    return output_path
