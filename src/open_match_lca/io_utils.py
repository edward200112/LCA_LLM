from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = require_exists(Path(path))
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"Invalid YAML config at {yaml_path}: {config}")
    return config


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    ensure_directory(out_path.parent)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    input_path = require_exists(Path(path))
    return json.loads(input_path.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = require_exists(Path(path))
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    ensure_directory(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_tabular_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported tabular file format: {path}")


def read_tabular_dir(directory: str | Path) -> pd.DataFrame:
    dir_path = require_exists(Path(directory))
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Expected a directory, got: {dir_path}")
    files = sorted(
        path
        for path in dir_path.iterdir()
        if path.suffix.lower() in {".csv", ".parquet", ".json", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No supported input files found in: {dir_path}")
    frames = [read_tabular_path(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    out_path = Path(path)
    ensure_directory(out_path.parent)
    frame.to_parquet(out_path, index=False)


def get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
