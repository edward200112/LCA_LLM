from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from open_match_lca.eval.eval_retrieval import compute_retrieval_metrics
from open_match_lca.io_utils import dump_json, ensure_directory, read_jsonl, write_jsonl, write_parquet
from open_match_lca.torch_utils import resolve_torch_device

try:
    from datasets import Dataset
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from transformers import TrainerCallback
except ImportError as exc:  # pragma: no cover
    Dataset = None
    InputExample = None
    CrossEncoder = None
    CrossEncoderTrainingArguments = None
    CrossEncoderTrainer = None
    TrainerCallback = None
    _RERANK_IMPORT_ERROR = exc
else:
    _RERANK_IMPORT_ERROR = None

if TYPE_CHECKING:
    from logging import Logger


REQUIRED_RERANK_PAIR_COLUMNS = ["query_text", "candidate_text", "label"]


class PairScorer(Protocol):
    def predict(
        self,
        inputs: list[list[str]],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RerankerArtifacts:
    model_dir: Path
    checkpoint_dir: Path
    dev_pair_scores_path: Path
    train_pair_count: int
    dev_pair_count: int


def validate_reranker_pair_frame(frame: pd.DataFrame, frame_name: str) -> None:
    missing = [column for column in REQUIRED_RERANK_PAIR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError(f"{frame_name} is empty.")


def _corpus_text_lookup(corpus: pd.DataFrame) -> dict[str, str]:
    if "naics_code" not in corpus.columns or "naics_text" not in corpus.columns:
        raise ValueError(
            f"Corpus must contain naics_code and naics_text columns. Available columns: {list(corpus.columns)}"
        )
    return (
        corpus.drop_duplicates(subset=["naics_code"])
        .set_index("naics_code")["naics_text"]
        .astype(str)
        .to_dict()
    )


def build_reranker_pairs_from_run(
    retrieval_records: list[dict],
    corpus: pd.DataFrame,
    top_k: int = 50,
) -> pd.DataFrame:
    corpus_lookup = _corpus_text_lookup(corpus)
    rows: list[dict] = []
    for record in retrieval_records:
        for rank, candidate in enumerate(record.get("candidates", [])[:top_k], start=1):
            naics_code = str(candidate.get("naics_code"))
            if naics_code not in corpus_lookup:
                raise ValueError(
                    f"Candidate NAICS code {naics_code} missing from corpus for reranker pair construction."
                )
            rows.append(
                {
                    "product_id": record.get("product_id"),
                    "gold_naics_code": record.get("gold_naics_code"),
                    "query_text": record.get("query_text", ""),
                    "candidate_text": corpus_lookup[naics_code],
                    "candidate_id": candidate.get("candidate_id", naics_code),
                    "naics_code": naics_code,
                    "initial_score": float(candidate.get("score", 0.0)),
                    "rank": rank,
                    "label": float(str(record.get("gold_naics_code")) == naics_code),
                }
            )
    frame = pd.DataFrame(rows)
    validate_reranker_pair_frame(frame, "reranker_pairs")
    return frame


def _build_input_examples(frame: pd.DataFrame) -> list[InputExample]:
    if InputExample is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is not installed. Install full project dependencies "
            "before training a reranker."
        ) from _RERANK_IMPORT_ERROR
    return [
        InputExample(
            texts=[str(row.query_text), str(row.candidate_text)],
            label=float(row.label),
        )
        for row in frame.itertuples(index=False)
    ]


def _build_trainer_dataset(frame: pd.DataFrame) -> Dataset:
    if Dataset is None:  # pragma: no cover
        raise RuntimeError(
            "datasets is not installed. Install full project dependencies before training a reranker."
        ) from _RERANK_IMPORT_ERROR
    validate_reranker_pair_frame(frame, "reranker_dataset")
    return Dataset.from_dict(
        {
            "sentence_0": frame["query_text"].astype(str).tolist(),
            "sentence_1": frame["candidate_text"].astype(str).tolist(),
            "label": frame["label"].astype(float).tolist(),
        }
    )


class EpochMetricsCallback(TrainerCallback):
    def __init__(
        self,
        cross_encoder: CrossEncoder,
        dev_pairs: pd.DataFrame,
        epoch_metrics_history: list[dict[str, float | int]],
        top_k: int,
        batch_size: int,
        logger: "Logger" | None = None,
    ) -> None:
        self.cross_encoder = cross_encoder
        self.dev_pairs = dev_pairs
        self.epoch_metrics_history = epoch_metrics_history
        self.top_k = top_k
        self.batch_size = batch_size
        self.logger = logger

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0) + 1
        if self.logger is not None:
            self.logger.info(
                "reranker_epoch_started",
                extra={
                    "structured": {
                        "epoch": epoch,
                        "epochs": int(args.num_train_epochs),
                        "device": str(args.device),
                    }
                },
            )
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        dev_scored = self.dev_pairs.copy()
        dev_inputs = dev_scored[["query_text", "candidate_text"]].astype(str).values.tolist()
        dev_scored["rerank_score"] = self.cross_encoder.predict(
            dev_inputs,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        epoch_records = scored_pairs_to_retrieval_records(dev_scored, top_k=self.top_k)
        epoch_metrics = compute_retrieval_metrics(epoch_records)
        epoch_metrics["epoch"] = int(round(state.epoch or 0))
        self.epoch_metrics_history.append(epoch_metrics)
        if self.logger is not None:
            self.logger.info("reranker_epoch_completed", extra={"structured": {"metrics": epoch_metrics}})
        return control


def scored_pairs_to_retrieval_records(
    scored_pairs: pd.DataFrame,
    top_k: int,
    score_column: str = "rerank_score",
) -> list[dict]:
    required = {"product_id", "gold_naics_code", "query_text", "naics_code", score_column}
    missing = sorted(required - set(scored_pairs.columns))
    if missing:
        raise ValueError(
            f"Scored reranker pairs are missing required columns: {missing}. "
            f"Available columns: {list(scored_pairs.columns)}"
        )
    records: list[dict] = []
    grouped = scored_pairs.groupby(["product_id", "gold_naics_code", "query_text"], sort=False)
    for (product_id, gold_naics_code, query_text), group in grouped:
        sorted_group = group.sort_values(
            by=[score_column, "initial_score"],
            ascending=[False, False],
        ).head(top_k)
        candidates = [
            {
                "candidate_id": getattr(row, "candidate_id", row.naics_code),
                "naics_code": str(row.naics_code),
                "score": float(getattr(row, "initial_score", 0.0)),
                "rerank_score": float(getattr(row, score_column)),
            }
            for row in sorted_group.itertuples(index=False)
        ]
        records.append(
            {
                "product_id": product_id,
                "gold_naics_code": gold_naics_code,
                "query_text": query_text,
                "candidates": candidates,
            }
        )
    return records


def train_cross_encoder_reranker(
    train_pairs: pd.DataFrame,
    dev_pairs: pd.DataFrame,
    base_model: str,
    output_dir: str | Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_length: int,
    top_k: int = 10,
    device: str | None = None,
    logger: "Logger" | None = None,
    checkpoint_save_steps: int = 300,
    checkpoint_save_total_limit: int = 2,
) -> RerankerArtifacts:
    if (
        CrossEncoder is None
        or CrossEncoderTrainer is None
        or CrossEncoderTrainingArguments is None
        or TrainerCallback is None
        or Dataset is None
    ):  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers/datasets/transformers are not fully installed. Install full project dependencies "
            "before training a reranker."
        ) from _RERANK_IMPORT_ERROR

    validate_reranker_pair_frame(train_pairs, "train_pairs")
    validate_reranker_pair_frame(dev_pairs, "dev_pairs")

    output_root = Path(output_dir)
    ensure_directory(output_root)
    model_dir = output_root / "reranker_model"
    checkpoint_dir = output_root / "checkpoints"
    ensure_directory(checkpoint_dir)
    resolved_device = resolve_torch_device(device)
    model = CrossEncoder(base_model, num_labels=1, max_length=max_length, device=resolved_device)
    train_dataset = _build_trainer_dataset(train_pairs)
    steps_per_epoch = max(1, (len(train_pairs) + batch_size - 1) // batch_size)
    warmup_steps = max(0, int(steps_per_epoch * epochs * 0.1))
    epoch_metrics_history: list[dict[str, float | int]] = []
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        save_strategy="steps",
        save_steps=checkpoint_save_steps,
        save_total_limit=checkpoint_save_total_limit,
        logging_strategy="steps",
        logging_steps=max(1, min(50, checkpoint_save_steps)),
        eval_strategy="no",
        disable_tqdm=False,
        max_grad_norm=1.0,
        warmup_steps=warmup_steps,
        dataloader_num_workers=0,
        report_to=[],
    )
    callbacks = [
        EpochMetricsCallback(
            cross_encoder=model,
            dev_pairs=dev_pairs,
            epoch_metrics_history=epoch_metrics_history,
            top_k=top_k,
            batch_size=batch_size,
            logger=logger,
        )
    ]
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=model.tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    model.save(str(model_dir))
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise RuntimeError(
            f"Reranker training completed but final model directory was not saved correctly: {model_dir}"
        )

    dev_scored = dev_pairs.copy()
    dev_inputs = dev_scored[["query_text", "candidate_text"]].astype(str).values.tolist()
    dev_scored["rerank_score"] = model.predict(
        dev_inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    dev_pair_scores_path = output_root / "dev_pair_scores.parquet"
    write_parquet(dev_scored, dev_pair_scores_path)
    dump_json(
        {
            "train_pairs": int(len(train_pairs)),
            "dev_pairs": int(len(dev_pairs)),
            "positive_rate_train": float(train_pairs["label"].mean()),
            "positive_rate_dev": float(dev_pairs["label"].mean()),
            "device": resolved_device,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_save_steps": int(checkpoint_save_steps),
            "checkpoint_save_total_limit": int(checkpoint_save_total_limit),
        },
        output_root / "reranker_training_summary.json",
    )
    dump_json(epoch_metrics_history, output_root / "reranker_epoch_metrics.json")
    return RerankerArtifacts(
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        dev_pair_scores_path=dev_pair_scores_path,
        train_pair_count=int(len(train_pairs)),
        dev_pair_count=int(len(dev_pairs)),
    )


def load_pair_frame(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Reranker pair file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    elif suffix == ".jsonl":
        frame = pd.DataFrame(read_jsonl(file_path))
    elif suffix == ".json":
        frame = pd.read_json(file_path)
    elif suffix == ".csv":
        frame = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported reranker pair format: {file_path}")
    validate_reranker_pair_frame(frame, str(file_path))
    return frame


def rerank_retrieval_records(
    retrieval_records: list[dict],
    corpus: pd.DataFrame,
    model_name_or_path: str,
    batch_size: int = 8,
    top_k: int = 10,
    scorer: PairScorer | None = None,
    device: str | None = None,
    show_progress_bar: bool = True,
) -> list[dict]:
    scorer_obj: PairScorer
    if scorer is not None:
        scorer_obj = scorer
    else:
        if CrossEncoder is None:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Install full project dependencies "
                "before using reranking."
            ) from _RERANK_IMPORT_ERROR
        scorer_obj = CrossEncoder(model_name_or_path, num_labels=1, device=resolve_torch_device(device))

    corpus_lookup = _corpus_text_lookup(corpus)
    reranked_records: list[dict] = []
    iterator = retrieval_records
    if show_progress_bar:
        iterator = tqdm(retrieval_records, desc="Cross-encoder reranking", unit="query")
    for record in iterator:
        candidates = record.get("candidates", [])
        pair_inputs = []
        enriched = []
        for candidate in candidates:
            naics_code = str(candidate.get("naics_code"))
            if naics_code not in corpus_lookup:
                raise ValueError(
                    f"Candidate NAICS code {naics_code} missing from corpus for reranking."
                )
            pair_inputs.append([str(record.get("query_text", "")), corpus_lookup[naics_code]])
            enriched.append(
                {
                    **candidate,
                    "candidate_text": corpus_lookup[naics_code],
                    "initial_score": float(candidate.get("score", 0.0)),
                }
            )
        if pair_inputs:
            scores = scorer_obj.predict(
                pair_inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for candidate, score in zip(enriched, scores, strict=False):
                candidate["rerank_score"] = float(score)
            reranked_candidates = sorted(
                enriched,
                key=lambda item: (item["rerank_score"], item["initial_score"]),
                reverse=True,
            )[:top_k]
        else:
            reranked_candidates = []
        reranked_records.append(
            {
                "product_id": record.get("product_id"),
                "gold_naics_code": record.get("gold_naics_code"),
                "query_text": record.get("query_text", ""),
                "candidates": reranked_candidates,
            }
        )
    return reranked_records


def save_reranked_records(records: list[dict], path: str | Path) -> Path:
    output_path = Path(path)
    write_jsonl(records, output_path)
    return output_path
