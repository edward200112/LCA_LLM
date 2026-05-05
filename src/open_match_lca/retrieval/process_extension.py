from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from open_match_lca.data.openlca_hybrid import _parse_openlca_process_directory
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
        openlca_root = _find_openlca_export_root(input_path)
        if openlca_root is not None:
            return _normalize_process_frame(_parse_openlca_process_directory(openlca_root, "uslci"))
        return _normalize_process_frame(parse_uslci_jsonld(str(input_path)))
    return _normalize_process_frame(read_tabular_path(input_path))


def _find_openlca_export_root(path: Path) -> Path | None:
    if (path / "openlca.json").exists() and (path / "processes").is_dir():
        return path
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "openlca.json").exists() and (child / "processes").is_dir():
            return child
    return None


def _normalize_process_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy().fillna("")
    rename_map = {
        "process_category": "category_path",
        "location": "geography",
        "source_repo": "source_dataset",
        "retrieval_text": "process_text",
    }
    for source_column, target_column in rename_map.items():
        if source_column in normalized.columns and target_column not in normalized.columns:
            normalized[target_column] = normalized[source_column]
    if "source_dataset" not in normalized.columns:
        normalized["source_dataset"] = ""
    if "source_type" not in normalized.columns:
        normalized["source_type"] = normalized["source_dataset"]
    return normalized


def load_process_corpus(path: str | Path) -> pd.DataFrame:
    return load_uslci_processes(path)


def load_process_exchanges(path: str | Path) -> pd.DataFrame:
    return read_tabular_path(require_exists(Path(path)))


def load_process_items(path: str | Path) -> pd.DataFrame:
    return read_tabular_path(require_exists(Path(path)))


def _jsonable_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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
        "source_dataset": row.get("source_dataset"),
        "source_type": row.get("source_type"),
        "top_item_names": _jsonable_value(row.get("top_item_names")),
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


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _listify(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist() if str(item)]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [text]


def _normalize_item_recommendation_config(domain_config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = {
        "domain_profile": "",
        "enable_domain_rerank": False,
        "enable_domain_filter": False,
        "domain_filter_min_score": 0.0,
        "domain_keep_topn_per_bucket": 0,
    }
    if domain_config:
        normalized.update(domain_config)
    normalized["domain_profile"] = str(normalized.get("domain_profile", "") or "").strip()
    normalized["enable_domain_rerank"] = _coerce_bool(normalized.get("enable_domain_rerank"))
    normalized["enable_domain_filter"] = _coerce_bool(normalized.get("enable_domain_filter"))
    normalized["domain_filter_min_score"] = _coerce_float(normalized.get("domain_filter_min_score"), 0.0)
    normalized["domain_keep_topn_per_bucket"] = max(
        0,
        _coerce_int(normalized.get("domain_keep_topn_per_bucket"), 0),
    )
    supported_profiles = {"", "pv_glass"}
    if normalized["domain_profile"] not in supported_profiles:
        raise ValueError(
            f"Unsupported domain_profile: {normalized['domain_profile']}. "
            f"Supported values: {sorted(supported_profiles)}"
        )
    return normalized


PV_GLASS_MATERIAL_TERMS = [
    "glass",
    "low iron",
    "low-iron",
    "silica sand",
    "silica",
    "soda ash",
    "limestone",
    "dolomite",
    "feldspar",
    "cullet",
    "tin",
    "eva",
    "pvb",
    "encapsulant",
    "interlayer",
    "laminate",
]

PV_GLASS_PROCESS_TERMS = [
    "tempering",
    "tempered",
    "temper",
    "coating",
    "coated",
    "lamination",
    "laminate",
    "furnace",
    "melting",
    "melt",
    "annealing",
    "anneal",
    "float bath",
    "forming",
    "washing",
    "drying",
    "cutting",
    "edge finishing",
]

PV_GLASS_TRANSPORT_TERMS = [
    "transport",
    "truck",
    "freight",
    "rail",
    "shipping",
    "logistics",
    "train",
    "t*km",
]

PV_GLASS_PACKAGING_TERMS = [
    "packaging",
    "package",
    "pallet",
    "corrugated",
    "cardboard",
    "crate",
    "box",
    "wrapping",
    "wrap",
    "film",
]

PV_GLASS_ELECTRICITY_TERMS = [
    "electricity",
    "electric power",
    "electric, ac",
    "electricity, ac",
    "kwh",
]

PV_GLASS_HEAT_FUEL_TERMS = [
    "natural gas",
    "fuel",
    "diesel",
    "steam",
    "heat",
    "coal",
    "propane",
    "gasoline",
]

PV_GLASS_SERVICE_TERMS = [
    "service",
    "administrative",
    "office",
    "consulting",
    "insurance",
    "rental",
]

PV_GLASS_WASTE_TERMS = [
    "waste",
    "disposal",
    "treatment",
    "landfill",
    "incineration",
    "recycling",
]

PV_GLASS_PRIORITY_BUCKETS = ["material", "process"]

PROCESS_ITEM_RECOMMENDATION_OUTPUT_COLUMNS = [
    "product_id",
    "query_text",
    "item_rank",
    "standardized_item_key",
    "standardized_name",
    "canonical_unit",
    "recommendation_score",
    "support_count",
    "supporting_process_uuids",
    "supporting_process_names",
    "source_datasets",
    "aliases",
    "domain_profile",
    "item_class_lv1",
    "item_class_lv2",
    "base_score",
    "domain_adjusted_score",
    "domain_action",
    "domain_reason",
    "domain_is_retained",
]


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _item_search_text(row: pd.Series) -> str:
    parts = [
        str(row.get("standardized_name", "")),
        str(row.get("standardized_item_key", "")).replace("_", " "),
        " ".join(_listify(row.get("aliases"))),
        " ".join(_listify(row.get("source_datasets"))),
    ]
    return " ".join(part.lower() for part in parts if part).strip()


def _classify_pv_glass_item(row: pd.Series) -> tuple[str, str]:
    text = _item_search_text(row)
    if _contains_any(text, PV_GLASS_MATERIAL_TERMS):
        return "material", "glass_material"
    if _contains_any(text, PV_GLASS_PROCESS_TERMS):
        return "process", "process_relevant"
    if _contains_any(text, PV_GLASS_TRANSPORT_TERMS):
        return "transport", "generic_transport"
    if _contains_any(text, PV_GLASS_PACKAGING_TERMS):
        return "packaging", "packaging"
    if _contains_any(text, PV_GLASS_ELECTRICITY_TERMS):
        return "energy", "generic_electricity"
    if _contains_any(text, PV_GLASS_HEAT_FUEL_TERMS):
        return "energy", "generic_heat_or_fuel"
    if _contains_any(text, PV_GLASS_SERVICE_TERMS):
        return "service", "service_like"
    if _contains_any(text, PV_GLASS_WASTE_TERMS):
        return "waste", "waste_or_disposal"
    return "other", "other"


def _apply_domain_profile(
    frame: pd.DataFrame,
    domain_config: dict[str, Any],
) -> pd.DataFrame:
    working = frame.copy()
    working["domain_profile"] = str(domain_config.get("domain_profile", ""))
    working["base_score"] = working["recommendation_score"].astype(float)
    working["domain_adjusted_score"] = working["base_score"]
    working["domain_action"] = "none"
    working["domain_reason"] = ""
    working["domain_is_retained"] = True
    working["item_class_lv1"] = "other"
    working["item_class_lv2"] = "other"

    domain_profile = str(domain_config.get("domain_profile", ""))
    if not domain_profile:
        return working

    if domain_profile != "pv_glass":
        raise ValueError(f"Unsupported domain_profile: {domain_profile}")

    reasons_by_index: dict[int, list[str]] = {}
    for index, row in working.iterrows():
        item_class_lv1, item_class_lv2 = _classify_pv_glass_item(row)
        working.at[index, "item_class_lv1"] = item_class_lv1
        working.at[index, "item_class_lv2"] = item_class_lv2

        if not bool(domain_config.get("enable_domain_rerank")):
            continue

        score_delta = 0.0
        reasons: list[str] = []
        if item_class_lv1 == "material":
            score_delta += 0.25
            reasons.append("domain_prioritized_material")
        elif item_class_lv1 == "process":
            score_delta += 0.15
            reasons.append("domain_prioritized_process_relevant_item")
        elif item_class_lv1 == "transport":
            score_delta -= 0.35
            reasons.append("domain_demoted_generic_transport")
        elif item_class_lv1 == "packaging":
            score_delta -= 0.30
            reasons.append("domain_demoted_packaging")
        elif item_class_lv2 == "generic_electricity":
            score_delta -= 0.15
            reasons.append("domain_demoted_generic_electricity")
        elif item_class_lv2 == "generic_heat_or_fuel":
            score_delta -= 0.12
            reasons.append("domain_demoted_generic_heat_or_fuel")
        elif item_class_lv1 == "service":
            score_delta -= 0.35
            reasons.append("domain_demoted_service_like_item")
        elif item_class_lv1 == "waste":
            score_delta -= 0.20
            reasons.append("domain_demoted_waste_or_disposal_item")

        working.at[index, "domain_adjusted_score"] = float(row["base_score"]) + score_delta
        if reasons:
            reasons_by_index[index] = reasons
            working.at[index, "domain_action"] = "prioritized" if score_delta > 0 else "demoted"

    if bool(domain_config.get("enable_domain_filter")):
        min_score = float(domain_config.get("domain_filter_min_score", 0.0))
        for index, row in working.iterrows():
            item_class_lv1 = str(row["item_class_lv1"])
            adjusted_score = float(row["domain_adjusted_score"])
            support_count = int(row["support_count"])
            should_filter = False
            if adjusted_score < min_score and item_class_lv1 in {"transport", "packaging", "service"}:
                should_filter = True
            elif adjusted_score < min_score and item_class_lv1 in {"waste", "other"} and support_count <= 1:
                should_filter = True
            if should_filter:
                working.at[index, "domain_is_retained"] = False
                working.at[index, "domain_action"] = "filtered"
                reasons = reasons_by_index.get(index, [])
                reasons.append("domain_filtered_low_value_generic_item")
                reasons_by_index[index] = reasons

    for index, reasons in reasons_by_index.items():
        working.at[index, "domain_reason"] = ";".join(reasons)
    return working


def _sort_recommendation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        ["domain_adjusted_score", "support_count", "standardized_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _prioritize_buckets(
    frame: pd.DataFrame,
    *,
    keep_topn_per_bucket: int,
) -> pd.DataFrame:
    if frame.empty or keep_topn_per_bucket <= 0:
        return _sort_recommendation_frame(frame)

    sorted_frame = _sort_recommendation_frame(frame)
    selected_indices: list[int] = []
    for bucket in PV_GLASS_PRIORITY_BUCKETS:
        bucket_frame = sorted_frame.loc[sorted_frame["item_class_lv1"].eq(bucket)].head(keep_topn_per_bucket)
        for index in bucket_frame.index.tolist():
            if index not in selected_indices:
                selected_indices.append(index)
    for index in sorted_frame.index.tolist():
        if index not in selected_indices:
            selected_indices.append(index)
    return sorted_frame.loc[selected_indices].reset_index(drop=True)


def _finalize_recommendation_outputs(
    frame: pd.DataFrame,
    *,
    top_k: int,
    domain_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()

    retained = frame.loc[frame["domain_is_retained"]].copy()
    filtered = frame.loc[~frame["domain_is_retained"]].copy()
    retained = _prioritize_buckets(
        retained,
        keep_topn_per_bucket=int(domain_config.get("domain_keep_topn_per_bucket", 0)),
    )
    retained = retained.head(top_k).reset_index(drop=True)
    retained["item_rank"] = range(1, len(retained) + 1)

    filtered = _sort_recommendation_frame(filtered)
    filtered["item_rank"] = pd.Series([pd.NA] * len(filtered), dtype="Int64")
    audit = pd.concat([retained, filtered], ignore_index=True)
    return retained, audit


def recommend_process_items(
    retrieval_records: list[dict],
    process_exchanges_frame: pd.DataFrame,
    process_items_frame: pd.DataFrame,
    *,
    top_k: int = 10,
    domain_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    normalized_config = _normalize_item_recommendation_config(domain_config)
    required_exchange_cols = {
        "process_uuid",
        "process_name",
        "exchange_direction",
        "standardized_item_key",
        "is_quantitative_reference",
        "is_recommendable",
    }
    required_item_cols = {
        "standardized_item_key",
        "standardized_name",
        "canonical_unit",
        "source_datasets",
        "aliases",
    }
    missing_exchange = sorted(required_exchange_cols.difference(process_exchanges_frame.columns))
    if missing_exchange:
        raise ValueError(
            f"process_exchanges_frame must contain {sorted(required_exchange_cols)}. "
            f"Available columns: {list(process_exchanges_frame.columns)}"
        )
    missing_items = sorted(required_item_cols.difference(process_items_frame.columns))
    if missing_items:
        raise ValueError(
            f"process_items_frame must contain {sorted(required_item_cols)}. "
            f"Available columns: {list(process_items_frame.columns)}"
        )

    working_exchanges = process_exchanges_frame.copy()
    working_exchanges["standardized_item_key"] = working_exchanges["standardized_item_key"].astype(str)
    working_exchanges = working_exchanges.loc[working_exchanges["standardized_item_key"].ne("")].copy()
    working_exchanges["is_recommendable"] = working_exchanges["is_recommendable"].map(_coerce_bool)
    working_exchanges["is_quantitative_reference"] = working_exchanges["is_quantitative_reference"].map(_coerce_bool)
    working_exchanges = working_exchanges.loc[working_exchanges["is_recommendable"]].copy()
    process_to_exchanges = {
        process_uuid: group.to_dict(orient="records")
        for process_uuid, group in working_exchanges.groupby("process_uuid", sort=False)
    }

    item_lookup = {
        str(row["standardized_item_key"]): row
        for row in process_items_frame.to_dict(orient="records")
    }

    rows: list[dict] = []
    for record in retrieval_records:
        aggregates: dict[str, dict] = {}
        for rank, candidate in enumerate(record.get("candidates", [])[:top_k], start=1):
            process_uuid = str(candidate.get("process_uuid", ""))
            process_name = str(candidate.get("process_name", ""))
            exchanges = process_to_exchanges.get(process_uuid, [])
            if not exchanges:
                continue
            base_weight = 1.0 / rank
            for exchange in exchanges:
                item_key = str(exchange.get("standardized_item_key", ""))
                item_row = item_lookup.get(item_key)
                if not item_key or item_row is None:
                    continue
                exchange_direction = str(exchange.get("exchange_direction", ""))
                if exchange_direction == "input":
                    exchange_weight = base_weight
                elif _coerce_bool(exchange.get("is_quantitative_reference")):
                    exchange_weight = base_weight * 0.75
                else:
                    continue

                if item_key not in aggregates:
                    aggregates[item_key] = {
                        "product_id": record.get("product_id"),
                        "query_text": record.get("query_text", ""),
                        "standardized_item_key": item_key,
                        "standardized_name": item_row.get("standardized_name", ""),
                        "canonical_unit": item_row.get("canonical_unit", ""),
                        "source_datasets": _listify(item_row.get("source_datasets")),
                        "aliases": _listify(item_row.get("aliases")),
                        "recommendation_score": 0.0,
                        "supporting_process_uuids": [],
                        "supporting_process_names": [],
                    }
                aggregate = aggregates[item_key]
                aggregate["recommendation_score"] += exchange_weight
                if process_uuid and process_uuid not in aggregate["supporting_process_uuids"]:
                    aggregate["supporting_process_uuids"].append(process_uuid)
                if process_name and process_name not in aggregate["supporting_process_names"]:
                    aggregate["supporting_process_names"].append(process_name)

        query_frame = pd.DataFrame(
            [
                {
                    "product_id": item["product_id"],
                    "query_text": item["query_text"],
                    "item_rank": pd.NA,
                    "standardized_item_key": item["standardized_item_key"],
                    "standardized_name": item["standardized_name"],
                    "canonical_unit": item["canonical_unit"],
                    "recommendation_score": float(item["recommendation_score"]),
                    "support_count": len(item["supporting_process_uuids"]),
                    "supporting_process_uuids": item["supporting_process_uuids"],
                    "supporting_process_names": item["supporting_process_names"],
                    "source_datasets": item["source_datasets"],
                    "aliases": item["aliases"],
                }
                for item in aggregates.values()
            ]
        )
        if query_frame.empty:
            continue
        query_frame = _apply_domain_profile(query_frame, normalized_config)
        final_query_frame, _ = _finalize_recommendation_outputs(
            query_frame,
            top_k=top_k,
            domain_config=normalized_config,
        )
        rows.extend(final_query_frame.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame(columns=PROCESS_ITEM_RECOMMENDATION_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=PROCESS_ITEM_RECOMMENDATION_OUTPUT_COLUMNS)


def recommend_process_items_with_audit(
    retrieval_records: list[dict],
    process_exchanges_frame: pd.DataFrame,
    process_items_frame: pd.DataFrame,
    *,
    top_k: int = 10,
    domain_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_config = _normalize_item_recommendation_config(domain_config)
    required_exchange_cols = {
        "process_uuid",
        "process_name",
        "exchange_direction",
        "standardized_item_key",
        "is_quantitative_reference",
        "is_recommendable",
    }
    required_item_cols = {
        "standardized_item_key",
        "standardized_name",
        "canonical_unit",
        "source_datasets",
        "aliases",
    }
    missing_exchange = sorted(required_exchange_cols.difference(process_exchanges_frame.columns))
    if missing_exchange:
        raise ValueError(
            f"process_exchanges_frame must contain {sorted(required_exchange_cols)}. "
            f"Available columns: {list(process_exchanges_frame.columns)}"
        )
    missing_items = sorted(required_item_cols.difference(process_items_frame.columns))
    if missing_items:
        raise ValueError(
            f"process_items_frame must contain {sorted(required_item_cols)}. "
            f"Available columns: {list(process_items_frame.columns)}"
        )

    recommendation_frame = recommend_process_items(
        retrieval_records,
        process_exchanges_frame,
        process_items_frame,
        top_k=top_k,
        domain_config=normalized_config,
    )
    if recommendation_frame.empty:
        return recommendation_frame.copy(), recommendation_frame.copy()

    working_exchanges = process_exchanges_frame.copy()
    working_exchanges["standardized_item_key"] = working_exchanges["standardized_item_key"].astype(str)
    working_exchanges = working_exchanges.loc[working_exchanges["standardized_item_key"].ne("")].copy()
    working_exchanges["is_recommendable"] = working_exchanges["is_recommendable"].map(_coerce_bool)
    working_exchanges["is_quantitative_reference"] = working_exchanges["is_quantitative_reference"].map(_coerce_bool)
    working_exchanges = working_exchanges.loc[working_exchanges["is_recommendable"]].copy()
    process_to_exchanges = {
        process_uuid: group.to_dict(orient="records")
        for process_uuid, group in working_exchanges.groupby("process_uuid", sort=False)
    }
    item_lookup = {
        str(row["standardized_item_key"]): row
        for row in process_items_frame.to_dict(orient="records")
    }

    audit_rows: list[dict] = []
    for record in retrieval_records:
        aggregates: dict[str, dict] = {}
        for rank, candidate in enumerate(record.get("candidates", [])[:top_k], start=1):
            process_uuid = str(candidate.get("process_uuid", ""))
            process_name = str(candidate.get("process_name", ""))
            exchanges = process_to_exchanges.get(process_uuid, [])
            if not exchanges:
                continue
            base_weight = 1.0 / rank
            for exchange in exchanges:
                item_key = str(exchange.get("standardized_item_key", ""))
                item_row = item_lookup.get(item_key)
                if not item_key or item_row is None:
                    continue
                exchange_direction = str(exchange.get("exchange_direction", ""))
                if exchange_direction == "input":
                    exchange_weight = base_weight
                elif _coerce_bool(exchange.get("is_quantitative_reference")):
                    exchange_weight = base_weight * 0.75
                else:
                    continue
                if item_key not in aggregates:
                    aggregates[item_key] = {
                        "product_id": record.get("product_id"),
                        "query_text": record.get("query_text", ""),
                        "item_rank": pd.NA,
                        "standardized_item_key": item_key,
                        "standardized_name": item_row.get("standardized_name", ""),
                        "canonical_unit": item_row.get("canonical_unit", ""),
                        "recommendation_score": 0.0,
                        "support_count": 0,
                        "supporting_process_uuids": [],
                        "supporting_process_names": [],
                        "source_datasets": _listify(item_row.get("source_datasets")),
                        "aliases": _listify(item_row.get("aliases")),
                    }
                aggregate = aggregates[item_key]
                aggregate["recommendation_score"] += exchange_weight
                if process_uuid and process_uuid not in aggregate["supporting_process_uuids"]:
                    aggregate["supporting_process_uuids"].append(process_uuid)
                if process_name and process_name not in aggregate["supporting_process_names"]:
                    aggregate["supporting_process_names"].append(process_name)
                aggregate["support_count"] = len(aggregate["supporting_process_uuids"])

        query_frame = pd.DataFrame(list(aggregates.values()))
        if query_frame.empty:
            continue
        query_frame = _apply_domain_profile(query_frame, normalized_config)
        _, query_audit = _finalize_recommendation_outputs(
            query_frame,
            top_k=top_k,
            domain_config=normalized_config,
        )
        audit_rows.extend(query_audit.to_dict(orient="records"))

    audit_frame = pd.DataFrame(audit_rows, columns=PROCESS_ITEM_RECOMMENDATION_OUTPUT_COLUMNS)
    return recommendation_frame, audit_frame
