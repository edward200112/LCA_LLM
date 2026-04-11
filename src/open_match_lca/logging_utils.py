from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from open_match_lca.io_utils import ensure_directory, get_git_commit_hash


class JsonlHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        ensure_directory(path.parent)

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "structured", None)
        if isinstance(extra, dict):
            payload.update(extra)
        with self.path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(payload, ensure_ascii=False) + "\n")


def setup_run_logger(
    script_name: str,
    log_dir: Path,
    config_path: str | None = None,
    seed: int | None = None,
    dataset_version: str | None = None,
    sample_count: int | None = None,
) -> tuple[logging.Logger, str]:
    ensure_directory(log_dir)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    logger_name = f"{script_name}.{run_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    text_log = log_dir / f"{script_name}_{run_id}.log"
    jsonl_log = log_dir / f"{script_name}_{run_id}.jsonl"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(text_log, encoding="utf-8")
    file_handler.setFormatter(formatter)
    jsonl_handler = JsonlHandler(jsonl_log)

    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.addHandler(jsonl_handler)

    logger.info(
        "run_started",
        extra={
            "structured": {
                "run_id": run_id,
                "git_commit_hash": get_git_commit_hash(),
                "config_path": config_path,
                "seed": seed,
                "dataset_version": dataset_version,
                "sample_count": sample_count,
            }
        },
    )
    return logger, run_id


def log_final_metrics(logger: logging.Logger, metrics: dict[str, Any]) -> None:
    logger.info("run_completed", extra={"structured": {"final_metrics": metrics}})
