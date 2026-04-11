from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from open_match_lca.features.hierarchy_features import hierarchical_distance
from open_match_lca.regression.topk_factor_mixture import softmax
from open_match_lca.schemas import normalize_naics_code
from open_match_lca.uncertainty.classification_confidence import confidence_from_scores
from open_match_lca.uncertainty.conformal_regression import apply_conformal_interval


NON_FEATURE_COLUMNS = {
    "product_id",
    "gold_naics_code",
    "pred_naics_code",
    "factor_baseline",
    "y_true",
    "lower",
    "upper",
    "lower_conformal",
    "upper_conformal",
    "confidence",
    "retained",
    "retained_risk",
    "error",
    "correct",
}


@dataclass
class TextProjector:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return self.svd.transform(matrix)


def build_factor_lookup(epa_factors: pd.DataFrame) -> dict[str, float]:
    if "naics_code" not in epa_factors.columns or "factor_value" not in epa_factors.columns:
        raise ValueError(
            f"epa_factors must include naics_code and factor_value. Available columns: {list(epa_factors.columns)}"
        )
    grouped = epa_factors.groupby("naics_code")["factor_value"].mean()
    return {str(code): float(value) for code, value in grouped.items()}


def fit_text_projector(
    texts: list[str],
    pca_dim: int = 64,
    random_state: int = 13,
    max_features: int = 2048,
) -> tuple[TextProjector, np.ndarray]:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    n_components = min(pca_dim, max(1, matrix.shape[0] - 1), matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    projected = svd.fit_transform(matrix)
    return TextProjector(vectorizer=vectorizer, svd=svd), projected


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return float(value)


def _pairwise_hierarchy_distance(codes: list[str]) -> float:
    if len(codes) < 2:
        return 0.0
    distances = []
    for left_index in range(len(codes)):
        for right_index in range(left_index + 1, len(codes)):
            distances.append(hierarchical_distance(codes[left_index], codes[right_index]))
    return float(np.mean(distances)) if distances else 0.0


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [column for column in numeric_columns if column not in NON_FEATURE_COLUMNS]


def build_regression_feature_frame(
    retrieval_records: list[dict],
    products_frame: pd.DataFrame,
    epa_factors: pd.DataFrame,
    top_k: int = 5,
    pca_dim: int = 64,
    text_projector: TextProjector | None = None,
    fit_projector: bool = False,
    random_state: int = 13,
    include_hierarchy_features: bool = True,
) -> tuple[pd.DataFrame, TextProjector | None]:
    if "product_id" not in products_frame.columns or "text" not in products_frame.columns:
        raise ValueError(
            f"products_frame must include product_id and text. Available columns: {list(products_frame.columns)}"
        )
    factor_lookup = build_factor_lookup(epa_factors)
    product_lookup = products_frame.set_index("product_id").to_dict("index")

    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for record in retrieval_records:
        product_id = record.get("product_id")
        if product_id not in product_lookup:
            raise ValueError(f"product_id {product_id} from retrieval records not found in products frame")
        product_row = product_lookup[product_id]
        query_text = str(product_row.get("text", record.get("query_text", "")))
        texts.append(query_text)
        candidates = record.get("candidates", [])[:top_k]
        candidate_codes = [normalize_naics_code(candidate.get("naics_code")) for candidate in candidates]
        scores = [_safe_float(candidate.get("rerank_score", candidate.get("score", 0.0))) for candidate in candidates]
        if scores:
            probs = softmax(scores)
            confidence = confidence_from_scores(scores)
        else:
            probs = np.asarray([], dtype=float)
            confidence = {"top1_probability": 0.0, "top1_top2_margin": 0.0, "entropy": 0.0}

        factor_values = [
            factor_lookup[code]
            for code in candidate_codes
            if code in factor_lookup
        ]
        available_scores = [
            score
            for code, score in zip(candidate_codes, scores, strict=False)
            if code in factor_lookup
        ]
        available_probs = softmax(available_scores) if available_scores else np.asarray([], dtype=float)
        weighted_mean = (
            float(np.dot(np.asarray(factor_values, dtype=float), available_probs))
            if len(factor_values) > 0
            else 0.0
        )
        top1_code = candidate_codes[0] if candidate_codes else ""
        top2_code = candidate_codes[1] if len(candidate_codes) > 1 else ""
        feature_row: dict[str, Any] = {
            "product_id": product_id,
            "gold_naics_code": normalize_naics_code(product_row.get("gold_naics_code", record.get("gold_naics_code", ""))),
            "pred_naics_code": top1_code,
            "y_true": product_row.get("factor_value"),
            "text_len": int(product_row.get("text_len", len(query_text.split()))),
            "has_numeric_tokens": int(bool(product_row.get("has_numeric_tokens", False))),
            "numeric_token_count": int(product_row.get("numeric_token_count", 0)),
            "top1_probability": float(confidence["top1_probability"]),
            "top1_top2_margin": float(confidence["top1_top2_margin"]),
            "score_entropy": float(confidence["entropy"]),
            "candidate_count": int(len(candidates)),
            "retrieval_score_top1": float(scores[0]) if len(scores) > 0 else 0.0,
            "retrieval_score_top2": float(scores[1]) if len(scores) > 1 else 0.0,
            "retrieval_score_gap": float(scores[0] - scores[1]) if len(scores) > 1 else float(scores[0]) if scores else 0.0,
            "factor_weighted_mean": weighted_mean,
            "factor_weighted_std": float(np.std(factor_values)) if factor_values else 0.0,
            "factor_min": float(np.min(factor_values)) if factor_values else 0.0,
            "factor_max": float(np.max(factor_values)) if factor_values else 0.0,
            "topk_factor_count": int(len(factor_values)),
            "top1_factor_value": float(factor_lookup.get(top1_code, 0.0)),
            "top2_factor_value": float(factor_lookup.get(top2_code, 0.0)) if top2_code else 0.0,
        }
        if include_hierarchy_features:
            feature_row["unique_codes_2d"] = int(len({code[:2] for code in candidate_codes if code}))
            feature_row["unique_codes_4d"] = int(len({code[:4] for code in candidate_codes if code}))
            feature_row["top1_top2_hierarchy_distance"] = float(
                hierarchical_distance(top1_code, top2_code) if top2_code else 0.0
            )
            feature_row["avg_pairwise_hierarchy_distance"] = _pairwise_hierarchy_distance(candidate_codes)
        for index in range(top_k):
            code = candidate_codes[index] if index < len(candidate_codes) else ""
            score = scores[index] if index < len(scores) else 0.0
            prob = probs[index] if index < len(probs) else 0.0
            feature_row[f"topk_score_{index + 1}"] = float(score)
            feature_row[f"topk_prob_{index + 1}"] = float(prob)
            feature_row[f"topk_factor_{index + 1}"] = float(factor_lookup.get(code, 0.0))
        rows.append(feature_row)

    feature_frame = pd.DataFrame(rows)
    if fit_projector:
        text_projector, projected = fit_text_projector(texts, pca_dim=pca_dim, random_state=random_state)
    elif text_projector is not None:
        projected = text_projector.transform(texts)
    else:
        projected = np.zeros((len(texts), 0), dtype=float)

    for index in range(projected.shape[1]):
        feature_frame[f"text_pca_{index:02d}"] = projected[:, index]
    return feature_frame, text_projector


def save_regression_bundle(
    bundle: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir) / "regression_bundle.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return output_path


def load_regression_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Regression bundle not found: {bundle_path}")
    return joblib.load(bundle_path)


def predict_with_regression_bundle(
    retrieval_records: list[dict],
    products_frame: pd.DataFrame,
    epa_factors: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    text_projector = bundle.get("text_projector")
    feature_frame, _ = build_regression_feature_frame(
        retrieval_records=retrieval_records,
        products_frame=products_frame,
        epa_factors=epa_factors,
        top_k=int(bundle["metadata"]["top_k"]),
        pca_dim=int(bundle["metadata"]["pca_dim"]),
        text_projector=text_projector,
        fit_projector=False,
        random_state=int(bundle["metadata"]["seed"]),
        include_hierarchy_features=bool(bundle["metadata"].get("use_hierarchy_features", True)),
    )
    feature_columns = list(bundle["metadata"]["feature_columns"])
    x_frame = feature_frame.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0)
    x_array = np.ascontiguousarray(x_frame.to_numpy(dtype=np.float32))

    point_model = bundle["point_model"]
    lower_model = bundle["lower_model"]
    upper_model = bundle["upper_model"]
    point_preds = np.asarray(point_model.predict(x_array), dtype=float)
    lower_preds = np.asarray(lower_model.predict(x_array), dtype=float)
    upper_preds = np.asarray(upper_model.predict(x_array), dtype=float)
    lower_preds, upper_preds = np.minimum(lower_preds, upper_preds), np.maximum(lower_preds, upper_preds)

    conformal_qhat = float(bundle["metadata"].get("conformal_qhat", 0.0))
    conformal_intervals = apply_conformal_interval([0.0] * len(point_preds), conformal_qhat)

    pred_frame = feature_frame.copy()
    pred_frame["pred_factor_value"] = point_preds
    pred_frame["lower"] = lower_preds
    pred_frame["upper"] = upper_preds
    pred_frame["lower_conformal"] = pred_frame["lower"] - conformal_qhat
    pred_frame["upper_conformal"] = pred_frame["upper"] + conformal_qhat
    pred_frame["interval_width"] = pred_frame["upper_conformal"] - pred_frame["lower_conformal"]
    pred_frame["confidence"] = pred_frame["top1_probability"] / (pred_frame["interval_width"] + 1e-6)
    threshold = float(bundle["metadata"].get("abstention_threshold", 0.0))
    pred_frame["retained"] = pred_frame["confidence"] >= threshold
    if "y_true" in pred_frame.columns:
        pred_frame["error"] = (pred_frame["pred_factor_value"] - pred_frame["y_true"]).abs()
        pred_frame["correct"] = (
            (pred_frame["y_true"] >= pred_frame["lower_conformal"])
            & (pred_frame["y_true"] <= pred_frame["upper_conformal"])
        ).astype(float)
    _ = conformal_intervals
    return pred_frame
