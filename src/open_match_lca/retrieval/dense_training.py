from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from torch.utils.data import DataLoader

from open_match_lca.eval.eval_retrieval import compute_retrieval_metrics
from open_match_lca.io_utils import dump_json, ensure_directory, write_jsonl, write_parquet
from open_match_lca.retrieval.candidate_generation import dense_zero_shot_retrieve
from open_match_lca.torch_utils import resolve_torch_device

try:
    from sentence_transformers import InputExample, SentenceTransformer
    from sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking import (
        MultipleNegativesRankingLoss,
    )
except ImportError as exc:  # pragma: no cover
    SentenceTransformer = None
    InputExample = None
    MultipleNegativesRankingLoss = None
    _DENSE_TRAIN_IMPORT_ERROR = exc
else:
    _DENSE_TRAIN_IMPORT_ERROR = None

if TYPE_CHECKING:
    from logging import Logger


REQUIRED_PRODUCT_COLUMNS = ["product_id", "text", "gold_naics_code"]
REQUIRED_CORPUS_COLUMNS = ["naics_code", "naics_text"]


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(2, min(8, cpu_count // 2))


@dataclass(frozen=True)
class DenseTrainArtifacts:
    model_dir: Path
    train_pairs_path: Path
    dev_run_path: Path
    dev_metrics_path: Path
    index_dir: Path


def _require_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def _distinct_code_counts(frame: pd.DataFrame, group_col: str) -> dict[str, int]:
    counts = (
        frame.groupby(group_col)["gold_naics_code"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    return {str(key): int(value) for key, value in counts.items()}


def choose_hard_negative_bucket(
    naics_code: str,
    code4_counts: dict[str, int],
    code2_counts: dict[str, int],
) -> tuple[str, str]:
    code4 = str(naics_code)[:4]
    code2 = str(naics_code)[:2]
    if code4_counts.get(code4, 0) > 1:
        return "4_digit", code4
    if code2_counts.get(code2, 0) > 1:
        return "2_digit", code2
    return "global", "global"


def _round_robin_bucket(bucket_frame: pd.DataFrame) -> pd.DataFrame:
    groups = {
        str(code): deque(
            bucket_frame.loc[bucket_frame["gold_naics_code"] == code]
            .sort_values(["product_id", "text"])
            .to_dict("records")
        )
        for code in sorted(bucket_frame["gold_naics_code"].unique())
    }
    ordered_rows: list[dict] = []
    while any(groups.values()):
        for code in sorted(groups):
            if groups[code]:
                ordered_rows.append(groups[code].popleft())
    return pd.DataFrame(ordered_rows)


def build_dense_training_pairs(
    train_products: pd.DataFrame,
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(train_products, REQUIRED_PRODUCT_COLUMNS, "train_products")
    _require_columns(corpus, REQUIRED_CORPUS_COLUMNS, "naics_corpus")

    corpus_lookup = (
        corpus.drop_duplicates(subset=["naics_code"])
        .set_index("naics_code")["naics_text"]
        .to_dict()
    )
    missing_codes = sorted(
        set(train_products["gold_naics_code"].astype(str)) - set(map(str, corpus_lookup.keys()))
    )
    if missing_codes:
        raise ValueError(
            f"Training products contain gold NAICS codes missing from corpus: {missing_codes[:20]}"
        )

    pairs = train_products.loc[:, REQUIRED_PRODUCT_COLUMNS].copy()
    pairs["gold_naics_code"] = pairs["gold_naics_code"].astype(str)
    pairs["positive_text"] = pairs["gold_naics_code"].map(corpus_lookup)
    pairs["naics_code_2"] = pairs["gold_naics_code"].str[:2]
    pairs["naics_code_4"] = pairs["gold_naics_code"].str[:4]

    code4_counts = _distinct_code_counts(pairs, "naics_code_4")
    code2_counts = _distinct_code_counts(pairs, "naics_code_2")
    bucket_info = pairs["gold_naics_code"].map(
        lambda code: choose_hard_negative_bucket(str(code), code4_counts, code2_counts)
    )
    pairs["hard_negative_level"] = bucket_info.map(lambda item: item[0])
    pairs["hard_negative_bucket"] = bucket_info.map(lambda item: item[1])

    ordered_frames = []
    for bucket_name in sorted(pairs["hard_negative_bucket"].unique()):
        bucket_frame = pairs.loc[pairs["hard_negative_bucket"] == bucket_name].reset_index(drop=True)
        ordered_frames.append(_round_robin_bucket(bucket_frame))
    ordered = pd.concat(ordered_frames, ignore_index=True)
    ordered["pair_id"] = [f"pair_{index:08d}" for index in range(len(ordered))]
    return ordered[
        [
            "pair_id",
            "product_id",
            "text",
            "gold_naics_code",
            "positive_text",
            "naics_code_2",
            "naics_code_4",
            "hard_negative_level",
            "hard_negative_bucket",
        ]
    ]


def build_input_examples(train_pairs: pd.DataFrame) -> list[InputExample]:
    if InputExample is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is not installed. Install full project dependencies "
            "before running dense fine-tuning."
        ) from _DENSE_TRAIN_IMPORT_ERROR
    return [
        InputExample(
            guid=str(row.pair_id),
            texts=[str(row.text), str(row.positive_text)],
        )
        for row in train_pairs.itertuples(index=False)
    ]


def train_dense_model(
    train_products: pd.DataFrame,
    dev_products: pd.DataFrame,
    corpus: pd.DataFrame,
    encoder_name: str,
    output_dir: str | Path,
    index_dir: str | Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_length: int,
    top_k: int,
    logger: Logger | None = None,
    device: str | None = None,
) -> DenseTrainArtifacts:
    if SentenceTransformer is None or MultipleNegativesRankingLoss is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is not installed. Install full project dependencies "
            "before running dense fine-tuning."
        ) from _DENSE_TRAIN_IMPORT_ERROR

    output_root = Path(output_dir)
    ensure_directory(output_root)
    index_root = Path(index_dir)
    ensure_directory(index_root)

    train_pairs = build_dense_training_pairs(train_products, corpus)
    if train_pairs["gold_naics_code"].nunique() < 2:
        raise RuntimeError(
            "Dense fine-tuning requires at least two distinct NAICS-6 labels "
            "to create in-batch negatives."
        )

    train_pairs_path = output_root / "train_pairs.parquet"
    write_parquet(train_pairs, train_pairs_path)

    resolved_device = resolve_torch_device(device)
    model = SentenceTransformer(encoder_name, device=resolved_device)
    model.max_seq_length = int(max_length)
    train_examples = build_input_examples(train_pairs)
    dataloader_num_workers = _default_num_workers()
    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=resolved_device.startswith("cuda"),
        persistent_workers=dataloader_num_workers > 0,
    )
    train_loss = MultipleNegativesRankingLoss(model)

    warmup_steps = max(0, int(len(train_loader) * 0.1))
    if logger is not None:
        logger.info(
            "dense_training_started",
            extra={
                "structured": {
                    "encoder_name": encoder_name,
                    "device": resolved_device,
                    "train_pairs": len(train_pairs),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "dataloader_num_workers": dataloader_num_workers,
                    "warmup_steps": warmup_steps,
                }
            },
        )

    model_dir = output_root / "dense_model"
    epoch_metrics_history: list[dict[str, float | int]] = []
    for epoch_index in range(epochs):
        if logger is not None:
            logger.info(
                "dense_epoch_started",
                extra={
                    "structured": {
                        "epoch": epoch_index + 1,
                        "epochs": epochs,
                        "device": resolved_device,
                    }
                },
            )
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch_index == 0 else 0,
            optimizer_params={"lr": learning_rate},
            output_path=str(model_dir),
            save_best_model=False,
            show_progress_bar=True,
        )
        epoch_runs = dense_zero_shot_retrieve(
            dev_products,
            corpus,
            top_k=top_k,
            encoder_name=str(model_dir),
            batch_size=batch_size,
            encoder=model,
            index_dir=str(index_root),
            device=resolved_device,
            show_progress_bar=True,
        )
        epoch_metrics = compute_retrieval_metrics(epoch_runs)
        epoch_metrics["epoch"] = int(epoch_index + 1)
        epoch_metrics_history.append(epoch_metrics)
        if logger is not None:
            logger.info("dense_epoch_completed", extra={"structured": {"metrics": epoch_metrics}})
    model.save(str(model_dir))

    dev_runs = dense_zero_shot_retrieve(
        dev_products,
        corpus,
        top_k=top_k,
        encoder_name=str(model_dir),
        batch_size=batch_size,
        encoder=model,
        index_dir=str(index_root),
        device=resolved_device,
        show_progress_bar=True,
    )
    dev_run_path = output_root / "retrieval_topk_dev_dense_finetuned.jsonl"
    write_jsonl(dev_runs, dev_run_path)

    dev_metrics = compute_retrieval_metrics(dev_runs)
    dev_metrics["train_pairs"] = int(len(train_pairs))
    dev_metrics["train_unique_labels"] = int(train_pairs["gold_naics_code"].nunique())
    dev_metrics["hard_negative_buckets"] = int(train_pairs["hard_negative_bucket"].nunique())
    dev_metrics["device"] = resolved_device
    dev_metrics_path = output_root / "retrieval_metrics_dev_dense_finetuned.json"
    dump_json(dev_metrics, dev_metrics_path)
    dump_json(epoch_metrics_history, output_root / "retrieval_metrics_dev_dense_finetuned_epochs.json")
    if logger is not None:
        logger.info("dense_training_completed", extra={"structured": {"dev_metrics": dev_metrics}})

    return DenseTrainArtifacts(
        model_dir=model_dir,
        train_pairs_path=train_pairs_path,
        dev_run_path=dev_run_path,
        dev_metrics_path=dev_metrics_path,
        index_dir=index_root,
    )
