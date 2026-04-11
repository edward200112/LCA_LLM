from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="open_match_lca_mpl_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def export_line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    output_path: str | Path,
    title: str,
) -> Path:
    out_path = Path(output_path)
    ensure_directory(out_path.parent)
    ax = frame.plot.line(x=x, y=y, title=title)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path)
    plt.close(ax.figure)
    return out_path


def export_histogram(
    values: pd.Series | np.ndarray | list[float],
    output_path: str | Path,
    title: str,
    bins: int = 20,
) -> Path:
    out_path = Path(output_path)
    ensure_directory(out_path.parent)
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def export_calibration_plot(
    frame: pd.DataFrame,
    confidence_col: str,
    correctness_col: str,
    output_path: str | Path,
    title: str,
    bins: int = 10,
) -> Path:
    out_path = Path(output_path)
    ensure_directory(out_path.parent)
    bucketed = frame.copy()
    bucketed["bin"] = pd.cut(bucketed[confidence_col], bins=bins, include_lowest=True)
    summary = (
        bucketed.groupby("bin", observed=False)
        .agg(mean_confidence=(confidence_col, "mean"), mean_correctness=(correctness_col, "mean"))
        .dropna()
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots()
    ax.plot(summary["mean_confidence"], summary["mean_correctness"], marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Observed correctness")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
