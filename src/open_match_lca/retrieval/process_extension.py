from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from open_match_lca.data.parse_uslci_jsonld import parse_uslci_jsonld
from open_match_lca.io_utils import read_tabular_path, require_exists
from open_match_lca.retrieval.bm25_retriever import BM25Retriever
from open_match_lca.retrieval.candidate_generation import lexical_overlap_score
from open_match_lca.retrieval.dense_retriever import DenseRetriever

try:
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError as exc:  # pragma: no cover
    CrossEncoder = None
    _PROCESS_RERANK_IMPORT_ERROR = exc
else:
    _PROCESS_RERANK_IMPORT_ERROR = None


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


def load_uslci_processes(path: str | Path) -> pd.DataFrame:
    input_path = require_exists(Path(path))
    if input_path.is_dir():
        return parse_uslci_jsonld(str(input_path))
    return read_tabular_path(input_path)


def _candidate_row_to_payload(row: pd.Series, score: float) -> dict:
    return {
        "candidate_id": row.get("process_uuid"),
        "process_uuid": row.get("process_uuid"),
        "process_name": row.get("process_name"),
        "category_path": row.get("category_path"),
        "geography": row.get("geography"),
        "reference_flow_name": row.get("reference_flow_name"),
        "reference_flow_unit": row.get("reference_flow_unit"),
        "process_text": row.get("process_text"),
        "score": float(score),
    }


def _uslci_prefilter_frame(product_row: pd.Series, uslci_frame: pd.DataFrame) -> pd.DataFrame:
    if "naics_code_2" not in uslci_frame.columns:
        return uslci_frame
    pred_code = str(product_row.get("pred_naics_code") or product_row.get("gold_naics_code") or "")
    if not pred_code:
        return uslci_frame
    code2 = pred_code[:2]
    filtered = uslci_frame.loc[uslci_frame["naics_code_2"].astype(str) == code2].reset_index(drop=True)
    return filtered if not filtered.empty else uslci_frame


def retrieve_process_candidates(
    products_frame: pd.DataFrame,
    uslci_frame: pd.DataFrame,
    retriever_ckpt: str,
    top_k: int = 10,
    prefilter_by_naics: bool = False,
    batch_size: int = 8,
) -> list[dict]:
    required_product_cols = {"product_id", "text"}
    required_uslci_cols = {"process_uuid", "process_name", "process_text"}
    if not required_product_cols.issubset(products_frame.columns):
        raise ValueError(
            f"products_frame must contain {sorted(required_product_cols)}. "
            f"Available columns: {list(products_frame.columns)}"
        )
    if not required_uslci_cols.issubset(uslci_frame.columns):
        raise ValueError(
            f"uslci_frame must contain {sorted(required_uslci_cols)}. "
            f"Available columns: {list(uslci_frame.columns)}"
        )

    outputs: list[dict] = []
    if retriever_ckpt == "bm25":
        if not prefilter_by_naics:
            retriever = BM25Retriever(uslci_frame["process_text"].astype(str).tolist())
            for row in products_frame.itertuples(index=False):
                hits = retriever.search(str(row.text), top_k=top_k)
                candidates = []
                for index, score in hits:
                    candidate_row = uslci_frame.iloc[index]
                    combined_score = float(score) + 1000.0 * lexical_overlap_score(
                        str(row.text), str(candidate_row["process_text"])
                    )
                    candidates.append(_candidate_row_to_payload(candidate_row, combined_score))
                candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]
                outputs.append(
                    {
                        "product_id": row.product_id,
                        "query_text": row.text,
                        "gold_process_uuid": getattr(row, "gold_process_uuid", None),
                        "candidates": candidates,
                    }
                )
            return outputs

        for _, product_row in products_frame.iterrows():
            candidate_frame = _uslci_prefilter_frame(product_row, uslci_frame)
            retriever = BM25Retriever(candidate_frame["process_text"].astype(str).tolist())
            hits = retriever.search(str(product_row["text"]), top_k=top_k)
            candidates = []
            for index, score in hits:
                candidate_row = candidate_frame.iloc[index]
                combined_score = float(score) + 1000.0 * lexical_overlap_score(
                    str(product_row["text"]), str(candidate_row["process_text"])
                )
                candidates.append(_candidate_row_to_payload(candidate_row, combined_score))
            candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]
            outputs.append(
                {
                    "product_id": product_row["product_id"],
                    "query_text": product_row["text"],
                    "gold_process_uuid": product_row.get("gold_process_uuid"),
                    "candidates": candidates,
                }
            )
        return outputs

    if prefilter_by_naics:
        for _, product_row in products_frame.iterrows():
            candidate_frame = _uslci_prefilter_frame(product_row, uslci_frame)
            retriever = DenseRetriever(
                corpus_texts=candidate_frame["process_text"].astype(str).tolist(),
                encoder_name=retriever_ckpt,
                batch_size=batch_size,
            )
            hits = retriever.search(str(product_row["text"]), top_k=top_k)
            candidates = [
                _candidate_row_to_payload(candidate_frame.iloc[hit.index], hit.score)
                for hit in hits
            ]
            outputs.append(
                {
                    "product_id": product_row["product_id"],
                    "query_text": product_row["text"],
                    "gold_process_uuid": product_row.get("gold_process_uuid"),
                    "candidates": candidates,
                }
            )
        return outputs

    retriever = DenseRetriever(
        corpus_texts=uslci_frame["process_text"].astype(str).tolist(),
        encoder_name=retriever_ckpt,
        batch_size=batch_size,
    )
    batch_hits = retriever.search_batch(products_frame["text"].astype(str).tolist(), top_k=top_k)
    for product_row, hits in zip(products_frame.itertuples(index=False), batch_hits, strict=False):
        candidates = [
            _candidate_row_to_payload(uslci_frame.iloc[hit.index], hit.score)
            for hit in hits
        ]
        outputs.append(
            {
                "product_id": product_row.product_id,
                "query_text": product_row.text,
                "gold_process_uuid": getattr(product_row, "gold_process_uuid", None),
                "candidates": candidates,
            }
        )
    return outputs


def rerank_process_candidates(
    retrieval_records: list[dict],
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
                "before using process reranking."
            ) from _PROCESS_RERANK_IMPORT_ERROR
        scorer_obj = CrossEncoder(model_name_or_path, num_labels=1)

    reranked_records: list[dict] = []
    for record in retrieval_records:
        candidates = record.get("candidates", [])
        pair_inputs = [
            [str(record.get("query_text", "")), str(candidate.get("process_text", ""))]
            for candidate in candidates
        ]
        if pair_inputs:
            scores = scorer_obj.predict(
                pair_inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            enriched = []
            for candidate, score in zip(candidates, scores, strict=False):
                enriched.append(
                    {
                        **candidate,
                        "initial_score": float(candidate.get("score", 0.0)),
                        "rerank_score": float(score),
                    }
                )
            enriched = sorted(
                enriched,
                key=lambda item: (item["rerank_score"], item["initial_score"]),
                reverse=True,
            )[:top_k]
        else:
            enriched = []
        reranked_records.append(
            {
                "product_id": record.get("product_id"),
                "query_text": record.get("query_text", ""),
                "gold_process_uuid": record.get("gold_process_uuid"),
                "candidates": enriched,
            }
        )
    return reranked_records
