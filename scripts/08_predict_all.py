from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import ensure_directory, load_yaml, read_jsonl, require_exists, write_parquet, write_jsonl
from open_match_lca.logging_utils import setup_run_logger
from open_match_lca.regression.baseline_factor_lookup import (
    build_factor_lookup,
    top1_factor_lookup_predictions,
    topk_factor_mixture_predictions,
)
from open_match_lca.logging_utils import log_final_metrics
from open_match_lca.regression.predict_regression import (
    load_regression_bundle,
    predict_with_regression_bundle,
)
from open_match_lca.retrieval.rerank_cross_encoder import CrossEncoder, rerank_retrieval_records
from open_match_lca.torch_utils import resolve_torch_device

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def resolve_epa_factors_path(config: dict, split_path: Path) -> Path:
    configured = config.get("epa_factors_path")
    candidate_paths = []
    if configured:
        candidate_paths.append(Path(str(configured)))
    candidate_paths.append(split_path.parent.parent / "processed" / "epa_factors.parquet")
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve EPA factor parquet. Tried: {[str(path) for path in candidate_paths]}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_ckpt", required=True)
    parser.add_argument("--reranker_ckpt", required=False)
    parser.add_argument("--regressor_ckpt", required=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def _resolve_predict_batch_size(config: dict, resolved_device: str) -> int:
    base_batch_size = int(config.get("batch_size", 8))
    if "rerank_batch_size" in config:
        return int(config["rerank_batch_size"])
    if "predict_batch_size" in config:
        return int(config["predict_batch_size"])
    if not _is_cuda_device(resolved_device):
        return base_batch_size
    return max(base_batch_size, min(int(config.get("max_predict_batch_size", 128)), base_batch_size * 16))


def _resolve_torch_dtype(config: dict, resolved_device: str):
    if torch is None or not _is_cuda_device(resolved_device):
        return None
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bool(config.get("bf16", bf16_supported)):
        return torch.bfloat16
    if bool(config.get("fp16", True)):
        return torch.float16
    return None


def _enable_cuda_inference_speedups(resolved_device: str, config: dict) -> None:
    if torch is None or not _is_cuda_device(resolved_device):
        return
    if bool(config.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def _rerank_with_auto_batch(
    *,
    retrieval_records: list[dict],
    corpus: pd.DataFrame,
    reranker_ckpt: str,
    config: dict,
    logger,
) -> tuple[list[dict], int, str]:
    if CrossEncoder is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Install full project dependencies before reranking."
        )
    resolved_device = resolve_torch_device(config.get("device"))
    requested_batch_size = _resolve_predict_batch_size(config, resolved_device)
    batch_size = requested_batch_size
    _enable_cuda_inference_speedups(resolved_device, config)

    model_kwargs = {}
    torch_dtype = _resolve_torch_dtype(config, resolved_device)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    scorer = CrossEncoder(
        reranker_ckpt,
        num_labels=1,
        device=resolved_device,
        model_kwargs=model_kwargs or None,
    )

    while True:
        try:
            reranked = rerank_retrieval_records(
                retrieval_records=retrieval_records,
                corpus=corpus,
                model_name_or_path=reranker_ckpt,
                batch_size=batch_size,
                top_k=int(config.get("rerank_top_k", config.get("top_k", 10))),
                scorer=scorer,
                device=resolved_device,
                show_progress_bar=True,
            )
            return reranked, batch_size, resolved_device
        except RuntimeError as exc:
            message = str(exc).lower()
            if not _is_cuda_device(resolved_device) or "out of memory" not in message or batch_size <= 1:
                raise
            next_batch_size = max(1, batch_size // 2)
            if logger is not None:
                logger.warning(
                    "rerank_predict_oom_retry",
                    extra={
                        "structured": {
                            "device": resolved_device,
                            "failed_batch_size": batch_size,
                            "retry_batch_size": next_batch_size,
                        }
                    },
                )
            batch_size = next_batch_size
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("08_predict_all", LOGS_DIR, config_path=args.config)

    split_path = require_exists(Path(args.split_path))
    _ = require_exists(Path(args.corpus_path))
    retrieval_path = require_exists(Path(args.retriever_ckpt))
    split_frame = pd.read_parquet(split_path)
    retrieval_records = read_jsonl(retrieval_path)
    corpus = pd.read_parquet(require_exists(Path(args.corpus_path)))

    epa_path = require_exists(resolve_epa_factors_path(config, split_path))
    epa_factors = pd.read_parquet(epa_path)
    factor_lookup = build_factor_lookup(epa_factors)

    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    split_name = split_path.stem.replace(".parquet", "")
    regression_top_k = int(config.get("regression_top_k", 5))
    factor_baselines = list(config.get("factor_baselines", ["top1_factor_lookup", "topk_factor_mixture"]))
    rerank_enabled = bool(config.get("whether_rerank", False)) and bool(args.reranker_ckpt)
    rerank_top_k = int(config.get("rerank_top_k", config.get("top_k", 10)))
    prediction_records = retrieval_records

    if rerank_enabled:
        reranked, effective_rerank_batch_size, resolved_device = _rerank_with_auto_batch(
            retrieval_records=retrieval_records,
            corpus=corpus,
            reranker_ckpt=args.reranker_ckpt,
            config=config,
            logger=logger,
        )
        rerank_output = output_dir / f"retrieval_topk_{split_name}_reranked.jsonl"
        write_jsonl(reranked, rerank_output)
        logger.info(
            "reranked_run_written",
            extra={
                "structured": {
                    "output_path": str(rerank_output),
                    "device": resolved_device,
                    "rerank_top_k": rerank_top_k,
                    "effective_rerank_batch_size": effective_rerank_batch_size,
                }
            },
        )
        prediction_records = reranked
    else:
        effective_rerank_batch_size = None

    written = 0
    for baseline_name in factor_baselines:
        if baseline_name == "top1_factor_lookup":
            preds = top1_factor_lookup_predictions(
                prediction_records,
                factor_lookup,
                split_frame,
                model_name=baseline_name,
            )
        elif baseline_name == "topk_factor_mixture":
            preds = topk_factor_mixture_predictions(
                prediction_records,
                factor_lookup,
                split_frame,
                top_k=regression_top_k,
                model_name=baseline_name,
            )
        else:
            raise RuntimeError(f"Unsupported factor baseline in config: {baseline_name}")
        output_path = output_dir / f"regression_preds_{split_name}_{baseline_name}.parquet"
        write_parquet(preds, output_path)
        logger.info("prediction_file_written", extra={"structured": {"output_path": str(output_path)}})
        written += 1

    if args.regressor_ckpt and bool(config.get("whether_regression", False)):
        regressor_bundle = load_regression_bundle(args.regressor_ckpt)
        regressor_preds = predict_with_regression_bundle(
            retrieval_records=prediction_records,
            products_frame=split_frame,
            epa_factors=epa_factors,
            bundle=regressor_bundle,
        )
        regressor_output = output_dir / f"regression_preds_{split_name}_lgbm_regressor.parquet"
        write_parquet(regressor_preds, regressor_output)
        logger.info("regressor_predictions_written", extra={"structured": {"output_path": str(regressor_output)}})
        written += 1
    log_final_metrics(
        logger,
        {
            "prediction_files": written,
            "rerank_enabled": rerank_enabled,
            "effective_rerank_batch_size": effective_rerank_batch_size,
        },
    )


if __name__ == "__main__":
    main()
