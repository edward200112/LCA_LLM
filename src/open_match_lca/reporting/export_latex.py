from __future__ import annotations

from pathlib import Path

import pandas as pd

from open_match_lca.io_utils import ensure_directory


def export_latex_table(frame: pd.DataFrame, output_dir: str | Path, stem: str) -> Path:
    out_root = Path(output_dir)
    ensure_directory(out_root)
    latex_path = out_root / f"{stem}.tex"
    latex_path.write_text(frame.to_latex(index=False), encoding="utf-8")
    return latex_path
