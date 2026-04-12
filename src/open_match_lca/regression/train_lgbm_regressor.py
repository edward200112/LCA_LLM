from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from open_match_lca.eval.eval_regression import compute_regression_metrics
from open_match_lca.io_utils import dump_json, ensure_directory, write_parquet
from open_match_lca.regression.predict_regression import (
    feature_columns_from_frame,
    save_regression_bundle,
)
from open_match_lca.uncertainty.abstention import learn_abstention_threshold
from open_match_lca.uncertainty.conformal_regression import conformal_quantile
from open_match_lca.uncertainty.regression_confidence import (
    apply_regression_confidence_calibrator,
    fit_regression_confidence_calibrator,
)

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover
    LGBMRegressor = None
    _LIGHTGBM_IMPORT_ERROR = exc
else:
    _LIGHTGBM_IMPORT_ERROR = None

if TYPE_CHECKING:
    from logging import Logger


@dataclass(frozen=True)
class LGBMArtifacts:
    bundle_path: Path
    dev_predictions_path: Path
    dev_metrics_path: Path


def _require_target(frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    if "y_true" not in frame.columns:
        raise ValueError(f"{frame_name} must contain y_true column.")
    filtered = frame.dropna(subset=["y_true"]).reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"{frame_name} has no non-null y_true values.")
    return filtered


def _fit_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    objective: str,
    alpha: float | None,
    random_state: int,
) -> LGBMRegressor:
    if LGBMRegressor is None:  # pragma: no cover
        raise RuntimeError(
            "lightgbm is not installed. Install full project dependencies before "
            "training the quantile regressor."
        ) from _LIGHTGBM_IMPORT_ERROR
    params = {
        "objective": objective,
        "random_state": random_state,
        "n_estimators": 120,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 1,
        "min_data_in_bin": 1,
        "n_jobs": 1,
        "verbosity": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    model = LGBMRegressor(**params)
    x_array = np.ascontiguousarray(train_x.to_numpy(dtype=np.float32))
    y_array = np.ascontiguousarray(train_y.to_numpy(dtype=np.float32))
    model.fit(x_array, y_array)
    return model


def train_lgbm_quantile_regressor(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    output_dir: str | Path,
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
    top_k: int = 5,
    pca_dim: int = 64,
    seed: int = 13,
    logger: Logger | None = None,
    text_projector: object | None = None,
    use_hierarchy_features: bool = True,
) -> LGBMArtifacts:
    train_frame = _require_target(train_frame, "train_frame")
    dev_frame = _require_target(dev_frame, "dev_frame")

    feature_columns = feature_columns_from_frame(train_frame)
    if not feature_columns:
        raise RuntimeError("No numeric feature columns found for LightGBM training.")

    train_x = train_frame.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0)
    train_y = train_frame["y_true"].astype(float)
    dev_x = dev_frame.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0)
    dev_y = dev_frame["y_true"].astype(float)

    lower_q, _, upper_q = quantiles
    point_model = _fit_model(train_x, train_y, objective="regression", alpha=None, random_state=seed)
    lower_model = _fit_model(train_x, train_y, objective="quantile", alpha=lower_q, random_state=seed)
    upper_model = _fit_model(train_x, train_y, objective="quantile", alpha=upper_q, random_state=seed)

    dev_x_array = np.ascontiguousarray(dev_x.to_numpy(dtype=np.float32))
    point_preds = np.asarray(point_model.predict(dev_x_array), dtype=float)
    lower_preds = np.asarray(lower_model.predict(dev_x_array), dtype=float)
    upper_preds = np.asarray(upper_model.predict(dev_x_array), dtype=float)
    lower_preds, upper_preds = np.minimum(lower_preds, upper_preds), np.maximum(lower_preds, upper_preds)

    nonconformity = np.maximum(lower_preds - dev_y.to_numpy(), dev_y.to_numpy() - upper_preds)
    nonconformity = np.maximum(nonconformity, 0.0)
    conformal_qhat = conformal_quantile(nonconformity.tolist(), alpha=1.0 - (upper_q - lower_q))

    dev_predictions = dev_frame.copy()
    dev_predictions["pred_factor_value"] = point_preds
    dev_predictions["lower"] = lower_preds
    dev_predictions["upper"] = upper_preds
    dev_predictions["lower_conformal"] = dev_predictions["lower"] - conformal_qhat
    dev_predictions["upper_conformal"] = dev_predictions["upper"] + conformal_qhat
    dev_predictions["interval_width"] = (
        dev_predictions["upper_conformal"] - dev_predictions["lower_conformal"]
    )
    dev_predictions["error"] = (
        dev_predictions["pred_factor_value"] - dev_predictions["y_true"]
    ).abs()
    dev_predictions["correct"] = (
        (dev_predictions["y_true"] >= dev_predictions["lower_conformal"])
        & (dev_predictions["y_true"] <= dev_predictions["upper_conformal"])
    ).astype(float)
    confidence_calibrator = fit_regression_confidence_calibrator(dev_predictions, error_col="error")
    dev_predictions["confidence"] = apply_regression_confidence_calibrator(
        dev_predictions,
        confidence_calibrator,
    )
    abstention_threshold = learn_abstention_threshold(
        dev_predictions,
        confidence_col="confidence",
        error_col="error",
    )
    dev_predictions["retained"] = dev_predictions["confidence"] >= abstention_threshold

    regression_metrics = compute_regression_metrics(
        dev_predictions["y_true"].tolist(),
        dev_predictions["pred_factor_value"].tolist(),
    )
    regression_metrics["conformal_qhat"] = float(conformal_qhat)
    regression_metrics["abstention_threshold"] = float(abstention_threshold)

    output_root = Path(output_dir)
    ensure_directory(output_root)
    dev_predictions_path = output_root / "dev_regression_predictions.parquet"
    dev_metrics_path = output_root / "dev_regression_metrics.json"
    write_parquet(dev_predictions, dev_predictions_path)
    dump_json(regression_metrics, dev_metrics_path)

    bundle = {
        "point_model": point_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "text_projector": text_projector,
        "metadata": {
            "feature_columns": feature_columns,
            "quantiles": list(quantiles),
            "conformal_qhat": float(conformal_qhat),
            "abstention_threshold": float(abstention_threshold),
            "confidence_calibrator": confidence_calibrator,
            "top_k": int(top_k),
            "pca_dim": int(pca_dim),
            "seed": int(seed),
            "use_hierarchy_features": bool(use_hierarchy_features),
        },
    }
    bundle_path = save_regression_bundle(bundle, output_root)

    if logger is not None:
        logger.info(
            "lgbm_regressor_trained",
            extra={
                "structured": {
                    "feature_columns": feature_columns,
                    "conformal_qhat": float(conformal_qhat),
                    "abstention_threshold": float(abstention_threshold),
                    "confidence_calibrator": confidence_calibrator,
                }
            },
        )

    return LGBMArtifacts(
        bundle_path=bundle_path,
        dev_predictions_path=dev_predictions_path,
        dev_metrics_path=dev_metrics_path,
    )
