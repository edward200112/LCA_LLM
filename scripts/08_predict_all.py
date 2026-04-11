from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import load_yaml, read_jsonl, require_exists, write_parquet, write_jsonl
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
from open_match_lca.retrieval.rerank_cross_encoder import rerank_retrieval_records


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
    split_name = split_path.stem.replace(".parquet", "")
    regression_top_k = int(config.get("regression_top_k", 5))
    factor_baselines = list(config.get("factor_baselines", ["top1_factor_lookup", "topk_factor_mixture"]))
    rerank_enabled = bool(config.get("whether_rerank", False)) and bool(args.reranker_ckpt)
    rerank_top_k = int(config.get("rerank_top_k", config.get("top_k", 10)))
    prediction_records = retrieval_records

    if rerank_enabled:
        reranked = rerank_retrieval_records(
            retrieval_records=retrieval_records,
            corpus=corpus,
            model_name_or_path=args.reranker_ckpt,
            batch_size=int(config.get("batch_size", 8)),
            top_k=rerank_top_k,
        )
        rerank_output = output_dir / f"retrieval_topk_{split_name}_reranked.jsonl"
        write_jsonl(reranked, rerank_output)
        logger.info("reranked_run_written", extra={"structured": {"output_path": str(rerank_output)}})
        prediction_records = reranked

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
    log_final_metrics(logger, {"prediction_files": written, "rerank_enabled": rerank_enabled})


if __name__ == "__main__":
    main()
