from __future__ import annotations

from pathlib import Path

import pandas as pd

from open_match_lca.io_utils import ensure_directory


def export_table(frame: pd.DataFrame, output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    out_root = Path(output_dir)
    ensure_directory(out_root)
    csv_path = out_root / f"{stem}.csv"
    json_path = out_root / f"{stem}.json"
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path
