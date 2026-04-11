from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_dir_str = str(src_dir)
    if src_dir.is_dir() and src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
