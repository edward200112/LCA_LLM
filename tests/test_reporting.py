from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_export_paper_tables_and_figures(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    metrics_dir = tmp_path / "metrics"
    pred_dir = tmp_path / "predictions"
    output_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    metrics_dir.mkdir()
    pred_dir.mkdir()

    (metrics_dir / "retrieval_metrics_random_stratified_bm25.json").write_text(
        json.dumps({"top1_accuracy": 0.5, "recall@10": 1.0, "mrr@10": 0.75}),
        encoding="utf-8",
    )
    (metrics_dir / "regression_metrics_random_stratified_lgbm_regressor.json").write_text(
        json.dumps({"mae": 0.2, "rmse": 0.3, "spearman": 0.8}),
        encoding="utf-8",
    )
    (metrics_dir / "uncertainty_metrics_random_stratified_lgbm_regressor.json").write_text(
        json.dumps({"empirical_coverage": 0.9, "average_interval_width": 0.4, "abstention_gain": 0.1}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "y_true": 1.0,
                "pred_factor_value": 1.1,
                "lower_conformal": 0.8,
                "upper_conformal": 1.2,
                "confidence": 0.9,
                "error": 0.1,
                "correct": 1.0,
            },
            {
                "y_true": 2.0,
                "pred_factor_value": 1.7,
                "lower_conformal": 1.5,
                "upper_conformal": 2.3,
                "confidence": 0.6,
                "error": 0.3,
                "correct": 1.0,
            },
        ]
    ).to_parquet(pred_dir / "regression_preds_random_stratified_lgbm_regressor.parquet", index=False)
    products_path = tmp_path / "products.parquet"
    pd.DataFrame(
        [
            {"gold_naics_code": "337127"},
            {"gold_naics_code": "337127"},
            {"gold_naics_code": "335139"},
        ]
    ).to_parquet(products_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "12_export_paper_tables.py"),
            "--metrics_dir",
            str(metrics_dir),
            "--output_dir",
            str(output_dir),
            "--format",
            "both",
            "--pred_dir",
            str(pred_dir),
            "--products_path",
            str(products_path),
        ],
        check=True,
    )

    assert (output_dir / "main_results_table.csv").exists()
    assert (output_dir / "uncertainty_table.tex").exists()
    assert (figures_dir / "retrieval_performance_bar_chart.png").exists()
    assert (figures_dir / "class_distribution.png").exists()
