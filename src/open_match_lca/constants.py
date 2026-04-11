from __future__ import annotations

from pathlib import Path

PACKAGE_NAME = "open_match_lca"
DEFAULT_SEEDS = [13, 42, 3407]
TEXT_SEPARATOR = " [SEP] "
RRF_K = 60

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = REPORTS_DIR / "logs"
METRICS_DIR = REPORTS_DIR / "metrics"
