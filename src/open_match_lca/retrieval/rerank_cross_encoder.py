from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from open_match_lca.io_utils import dump_json, ensure_directory, read_jsonl, write_jsonl, write_parquet

try:
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError as exc:  # pragma: no cover
    InputExample = None
    CrossEncoder = None
    _RERANK_IMPORT_ERROR = exc
else:
    _RERANK_IMPORT_ERROR = None


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


def train_cross_encoder_reranker(
    train_pairs: pd.DataFrame,
    dev_pairs: pd.DataFrame,
    base_model: str,
    output_dir: str | Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_length: int,
) -> RerankerArtifacts:
    if CrossEncoder is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is not installed. Install full project dependencies "
            "before training a reranker."
        ) from _RERANK_IMPORT_ERROR

    validate_reranker_pair_frame(train_pairs, "train_pairs")
    validate_reranker_pair_frame(dev_pairs, "dev_pairs")

    output_root = Path(output_dir)
    ensure_directory(output_root)
    model_dir = output_root / "reranker_model"

    train_loader = DataLoader(
        _build_input_examples(train_pairs),
        batch_size=batch_size,
        shuffle=True,
    )
    model = CrossEncoder(base_model, num_labels=1, max_length=max_length)
    warmup_steps = max(0, int(len(train_loader) * max(epochs, 1) * 0.1))
    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(model_dir),
        save_best_model=False,
        show_progress_bar=False,
    )

    dev_scored = dev_pairs.copy()
    dev_inputs = dev_scored[["query_text", "candidate_text"]].astype(str).values.tolist()
    dev_scored["rerank_score"] = model.predict(
        dev_inputs,
        batch_size=batch_size,
        show_progress_bar=False,
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
        },
        output_root / "reranker_training_summary.json",
    )
    return RerankerArtifacts(
        model_dir=model_dir,
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
        scorer_obj = CrossEncoder(model_name_or_path, num_labels=1)

    corpus_lookup = _corpus_text_lookup(corpus)
    reranked_records: list[dict] = []
    for record in retrieval_records:
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
