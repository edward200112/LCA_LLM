from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.eval.significance import paired_significance_tests, summarize_metric_by_model
from open_match_lca.io_utils import dump_json, ensure_directory
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metric_file_glob", required=True)
    parser.add_argument("--primary_model", required=True)
    parser.add_argument("--baseline_models", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--model_level", type=int, default=0, help="Relative path level under runs_dir used as model name.")
    return parser


def _load_metric_frame(runs_dir: Path, metric_file_glob: str, model_level: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(runs_dir.glob(f"**/{metric_file_glob}")):
        if "seed_" not in path.as_posix():
            continue
        relative = path.relative_to(runs_dir)
        parts = relative.parts
        model_name = parts[model_level] if len(parts) > model_level else parts[0]
        seed = None
        for part in parts:
            if part.startswith("seed_"):
                seed = int(part.split("_", maxsplit=1)[1])
                break
        if seed is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        row = {"model": model_name, "seed": seed, "metric_file": str(relative)}
        row.update(payload)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No metric files matched {metric_file_glob} under {runs_dir}")
    return frame


def main() -> None:
    args = build_parser().parse_args()
    logger, _ = setup_run_logger("14_run_significance", LOGS_DIR)
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)

    metric_frame = _load_metric_frame(runs_dir, args.metric_file_glob, args.model_level)
    summary = summarize_metric_by_model(metric_frame, "model", args.metrics)
    significance = paired_significance_tests(
        metric_frame,
        model_col="model",
        seed_col="seed",
        primary_model=args.primary_model,
        baseline_models=list(args.baseline_models),
        metric_names=list(args.metrics),
    )

    summary.to_csv(output_dir / "metric_summary.csv", index=False)
    significance.to_csv(output_dir / "paired_significance.csv", index=False)
    dump_json({"rows": summary.to_dict(orient="records")}, output_dir / "metric_summary.json")
    dump_json({"rows": significance.to_dict(orient="records")}, output_dir / "paired_significance.json")
    log_final_metrics(
        logger,
        {
            "summary_rows": int(len(summary)),
            "significance_rows": int(len(significance)),
        },
    )


if __name__ == "__main__":
    main()
