from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from open_match_lca.io_utils import ensure_directory


def export_bar_chart(frame: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str) -> Path:
    out_path = Path(output_path)
    ensure_directory(out_path.parent)
    ax = frame.plot.bar(x=x, y=y, legend=False, title=title)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path)
    plt.close(ax.figure)
    return out_path
