from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from open_match_lca.eval.eval_regression import compute_regression_metrics
from open_match_lca.eval.eval_retrieval import compute_retrieval_metrics
from open_match_lca.eval.eval_uncertainty import evaluate_uncertainty
from open_match_lca.io_utils import dump_json, ensure_directory, load_yaml, require_exists, write_jsonl, write_parquet
from open_match_lca.regression.baseline_factor_lookup import (
    build_factor_lookup as build_factor_lookup_simple,
    top1_factor_lookup_predictions,
    topk_factor_mixture_predictions,
)
from open_match_lca.regression.predict_regression import (
    build_regression_feature_frame,
    load_regression_bundle,
    predict_with_regression_bundle,
)
from open_match_lca.regression.train_lgbm_regressor import train_lgbm_quantile_regressor
from open_match_lca.retrieval.candidate_generation import bm25_retrieve, dense_zero_shot_retrieve
from open_match_lca.retrieval.hybrid_rrf import reciprocal_rank_fusion
from open_match_lca.retrieval.process_extension import load_uslci_processes, retrieve_process_candidates
from open_match_lca.retrieval.rerank_cross_encoder import (
    build_reranker_pairs_from_run,
    rerank_retrieval_records,
    train_cross_encoder_reranker,
)
from open_match_lca.retrieval.dense_training import train_dense_model


DEFAULT_ABLATIONS = [
    "bm25_only",
    "dense_only",
    "hybrid_no_rerank",
    "hybrid_with_rerank",
    "top1_factor_only",
    "topk_factor_mixture",
    "regressor_off",
    "regressor_on",
    "hierarchy_features_off",
    "uncertainty_off",
    "process_extension_off",
    "process_extension_on",
]


@dataclass(frozen=True)
class PipelinePaths:
    train_path: Path
    dev_path: Path
    test_path: Path
    corpus_path: Path
    epa_path: Path
    uslci_path: Path | None
    run_dir: Path
    metrics_dir: Path
    temp_config_dir: Path


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "bm25_only": {"retrieval_mode": "bm25", "whether_rerank": False, "whether_regression": False},
    "dense_only": {"retrieval_mode": "dense", "whether_rerank": False, "whether_regression": False},
    "hybrid_no_rerank": {"retrieval_mode": "hybrid", "whether_rerank": False},
    "hybrid_with_rerank": {"retrieval_mode": "hybrid", "whether_rerank": True},
    "top1_factor_only": {"whether_regression": False, "factor_baselines": ["top1_factor_lookup"]},
    "topk_factor_mixture": {"whether_regression": False, "factor_baselines": ["topk_factor_mixture"]},
    "regressor_off": {"whether_regression": False},
    "regressor_on": {"whether_regression": True},
    "hierarchy_features_off": {"use_hierarchy_features": False},
    "uncertainty_off": {"whether_uncertainty": False},
    "process_extension_off": {"whether_process_extension": False},
    "process_extension_on": {"whether_process_extension": True},
}


def apply_ablation(base_config: dict[str, Any], ablation_name: str) -> dict[str, Any]:
    if ablation_name not in ABLATION_OVERRIDES:
        raise RuntimeError(f"Unsupported ablation: {ablation_name}")
    updated = copy.deepcopy(base_config)
    _deep_update(updated, copy.deepcopy(ABLATION_OVERRIDES[ablation_name]))
    updated["ablation_name"] = ablation_name
    return updated


def materialize_ablation_configs(
    exp_config_path: str | Path,
    output_dir: str | Path,
    ablations: list[str] | None = None,
) -> list[Path]:
    base_config = load_yaml(exp_config_path)
    output_root = Path(output_dir)
    ensure_directory(output_root)
    selected = ablations or DEFAULT_ABLATIONS
    paths = []
    for ablation in selected:
        config = apply_ablation(base_config, ablation)
        path = output_root / f"{ablation}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def resolve_pipeline_paths(config: dict[str, Any], output_dir: str | Path, seed: int) -> PipelinePaths:
    split_type = str(config.get("split_type", "random_stratified"))
    splits_dir = Path(str(config.get("splits_dir", "data/splits")))
    processed_dir = Path(str(config.get("processed_dir", "data/processed")))
    run_dir = Path(output_dir) / f"seed_{seed}"
    metrics_dir = run_dir / "metrics"
    temp_config_dir = run_dir / "configs"
    ensure_directory(metrics_dir)
    ensure_directory(temp_config_dir)
    uslci_path_value = config.get("uslci_path")
    uslci_path = Path(str(uslci_path_value)) if uslci_path_value else None
    return PipelinePaths(
        train_path=require_exists(splits_dir / f"{split_type}_train.parquet"),
        dev_path=require_exists(splits_dir / f"{split_type}_dev.parquet"),
        test_path=require_exists(splits_dir / f"{split_type}_test.parquet"),
        corpus_path=require_exists(Path(str(config.get("corpus_path", processed_dir / "naics_corpus.parquet")))),
        epa_path=require_exists(Path(str(config.get("epa_factors_path", processed_dir / "epa_factors.parquet")))),
        uslci_path=uslci_path if uslci_path is None else require_exists(uslci_path),
        run_dir=run_dir,
        metrics_dir=metrics_dir,
        temp_config_dir=temp_config_dir,
    )


def build_pipeline_manifest(config: dict[str, Any], seed: int, output_dir: str | Path) -> dict[str, Any]:
    paths = resolve_pipeline_paths(config, output_dir, seed)
    retrieval_mode = str(config.get("retrieval_mode", "hybrid"))
    steps = [
        {"name": "load_splits", "train": str(paths.train_path), "dev": str(paths.dev_path), "test": str(paths.test_path)},
        {"name": "bm25_baseline", "enabled": retrieval_mode in {"bm25", "hybrid"}},
        {"name": "dense_retrieval", "enabled": retrieval_mode in {"dense", "hybrid"}},
        {"name": "hybrid_rrf", "enabled": retrieval_mode == "hybrid"},
        {"name": "reranker", "enabled": bool(config.get("whether_rerank", False))},
        {"name": "factor_baselines", "enabled": True, "models": list(config.get("factor_baselines", ["top1_factor_lookup", "topk_factor_mixture"]))},
        {"name": "regressor", "enabled": bool(config.get("whether_regression", False))},
        {"name": "uncertainty", "enabled": bool(config.get("whether_uncertainty", True))},
        {"name": "process_extension", "enabled": bool(config.get("whether_process_extension", False))},
    ]
    return {
        "seed": seed,
        "output_dir": str(paths.run_dir),
        "ablation_name": config.get("ablation_name"),
        "steps": steps,
    }


def _fuse_runs(left_records: list[dict], right_records: list[dict], top_k: int) -> list[dict]:
    fused = []
    for left, right in zip(left_records, right_records, strict=False):
        fused.append(
            {
                "product_id": left["product_id"],
                "gold_naics_code": left["gold_naics_code"],
                "query_text": left["query_text"],
                "candidates": reciprocal_rank_fusion(
                    [left.get("candidates", []), right.get("candidates", [])],
                    top_k=top_k,
                ),
            }
        )
    return fused


def _save_metrics(run_records: list[dict], output_path: Path) -> None:
    dump_json(compute_retrieval_metrics(run_records), output_path)


def run_pipeline(config: dict[str, Any], seed: int, output_dir: str | Path, dry_run: bool = False) -> dict[str, Any]:
    paths = resolve_pipeline_paths(config, output_dir, seed)
    manifest = build_pipeline_manifest(config, seed, output_dir)
    dump_json(manifest, paths.run_dir / "pipeline_manifest.json")
    if dry_run:
        return manifest

    train = pd.read_parquet(paths.train_path)
    dev = pd.read_parquet(paths.dev_path)
    test = pd.read_parquet(paths.test_path)
    corpus = pd.read_parquet(paths.corpus_path)
    epa = pd.read_parquet(paths.epa_path)

    top_k = int(config.get("top_k", 10))
    batch_size = int(config.get("batch_size", 8))
    dense_encoder_name = str(config.get("dense_encoder_name", "all-MiniLM-L6-v2"))
    rerank_base_model = str(config.get("rerank_base_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    retrieval_mode = str(config.get("retrieval_mode", "hybrid"))
    device = config.get("device")

    bm25_runs: dict[str, list[dict]] = {}
    dense_runs: dict[str, list[dict]] = {}
    selected_runs: dict[str, list[dict]]

    if retrieval_mode in {"bm25", "hybrid"}:
        for split_name, frame in [("train", train), ("dev", dev), ("test", test)]:
            bm25_runs[split_name] = bm25_retrieve(frame, corpus, top_k=top_k)
            write_jsonl(
                bm25_runs[split_name],
                paths.run_dir / f"retrieval_topk_{split_name}_bm25.jsonl",
            )

    if retrieval_mode in {"dense", "hybrid"}:
        dense_artifacts = train_dense_model(
            train_products=train,
            dev_products=dev,
            corpus=corpus,
            encoder_name=dense_encoder_name,
            output_dir=paths.run_dir / "dense_training",
            index_dir=paths.run_dir / "indices" / "dense_finetuned",
            batch_size=batch_size,
            epochs=int(config.get("epochs", 1)),
            learning_rate=float(config.get("learning_rate", 2e-5)),
            max_length=int(config.get("max_length", 256)),
            top_k=top_k,
            device=device,
        )
        for split_name, frame in [("train", train), ("dev", dev), ("test", test)]:
            dense_runs[split_name] = dense_zero_shot_retrieve(
                frame,
                corpus,
                top_k=top_k,
                encoder_name=str(dense_artifacts.model_dir),
                batch_size=batch_size,
                device=device,
                show_progress_bar=True,
            )
            write_jsonl(
                dense_runs[split_name],
                paths.run_dir / f"retrieval_topk_{split_name}_dense_finetuned.jsonl",
            )

    if retrieval_mode == "bm25":
        selected_runs = bm25_runs
        selected_model_name = "bm25"
    elif retrieval_mode == "dense":
        selected_runs = dense_runs
        selected_model_name = "dense_finetuned"
    else:
        selected_runs = {}
        for split_name in ["train", "dev", "test"]:
            selected_runs[split_name] = _fuse_runs(bm25_runs[split_name], dense_runs[split_name], top_k=top_k)
            write_jsonl(
                selected_runs[split_name],
                paths.run_dir / f"retrieval_topk_{split_name}_hybrid.jsonl",
            )
        selected_model_name = "hybrid"

    if bool(config.get("whether_rerank", False)):
        train_pairs = build_reranker_pairs_from_run(selected_runs["train"], corpus, top_k=top_k)
        dev_pairs = build_reranker_pairs_from_run(selected_runs["dev"], corpus, top_k=top_k)
        rerank_artifacts = train_cross_encoder_reranker(
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            base_model=rerank_base_model,
            output_dir=paths.run_dir / "reranker",
            batch_size=batch_size,
            epochs=int(config.get("epochs", 1)),
            learning_rate=float(config.get("learning_rate", 2e-5)),
            max_length=int(config.get("max_length", 256)),
            top_k=int(config.get("rerank_top_k", top_k)),
            device=device,
        )
        for split_name in ["train", "dev", "test"]:
            selected_runs[split_name] = rerank_retrieval_records(
                selected_runs[split_name],
                corpus,
                model_name_or_path=str(rerank_artifacts.model_dir),
                batch_size=batch_size,
                top_k=int(config.get("rerank_top_k", top_k)),
                device=device,
                show_progress_bar=True,
            )
            write_jsonl(
                selected_runs[split_name],
                paths.run_dir / f"retrieval_topk_{split_name}_{selected_model_name}_reranked.jsonl",
            )
        selected_model_name = f"{selected_model_name}_reranked"

    _save_metrics(selected_runs["test"], paths.metrics_dir / f"retrieval_metrics_test_{selected_model_name}.json")

    factor_lookup = build_factor_lookup_simple(epa)
    for baseline_name in list(config.get("factor_baselines", ["top1_factor_lookup", "topk_factor_mixture"])):
        if baseline_name == "top1_factor_lookup":
            preds = top1_factor_lookup_predictions(selected_runs["test"], factor_lookup, test, baseline_name)
        elif baseline_name == "topk_factor_mixture":
            preds = topk_factor_mixture_predictions(
                selected_runs["test"],
                factor_lookup,
                test,
                top_k=int(config.get("regression_top_k", 5)),
                model_name=baseline_name,
            )
        else:
            raise RuntimeError(f"Unsupported factor baseline: {baseline_name}")
        pred_path = paths.run_dir / f"regression_preds_test_{baseline_name}.parquet"
        write_parquet(preds, pred_path)
        eval_frame = preds.dropna(subset=["y_true", "pred_factor_value"])
        if not eval_frame.empty:
            dump_json(
                compute_regression_metrics(eval_frame["y_true"].tolist(), eval_frame["pred_factor_value"].tolist()),
                paths.metrics_dir / f"regression_metrics_test_{baseline_name}.json",
            )

    if bool(config.get("whether_regression", False)):
        train_features, projector = build_regression_feature_frame(
            selected_runs["train"],
            train,
            epa,
            top_k=int(config.get("regression_top_k", 5)),
            pca_dim=int(config.get("pca_dim", 64)),
            fit_projector=True,
            random_state=seed,
            include_hierarchy_features=bool(config.get("use_hierarchy_features", True)),
        )
        dev_features, _ = build_regression_feature_frame(
            selected_runs["dev"],
            dev,
            epa,
            top_k=int(config.get("regression_top_k", 5)),
            pca_dim=int(config.get("pca_dim", 64)),
            text_projector=projector,
            fit_projector=False,
            random_state=seed,
            include_hierarchy_features=bool(config.get("use_hierarchy_features", True)),
        )
        test_features, _ = build_regression_feature_frame(
            selected_runs["test"],
            test,
            epa,
            top_k=int(config.get("regression_top_k", 5)),
            pca_dim=int(config.get("pca_dim", 64)),
            text_projector=projector,
            fit_projector=False,
            random_state=seed,
            include_hierarchy_features=bool(config.get("use_hierarchy_features", True)),
        )
        write_parquet(train_features, paths.run_dir / "train_regression_features.parquet")
        write_parquet(dev_features, paths.run_dir / "dev_regression_features.parquet")
        artifacts = train_lgbm_quantile_regressor(
            train_frame=train_features,
            dev_frame=dev_features,
            output_dir=paths.run_dir / "regressor",
            quantiles=tuple(config.get("quantiles", [0.1, 0.5, 0.9])),
            top_k=int(config.get("regression_top_k", 5)),
            pca_dim=int(config.get("pca_dim", 64)),
            seed=seed,
            text_projector=projector,
        )
        bundle = load_regression_bundle(artifacts.bundle_path)
        reg_preds = predict_with_regression_bundle(selected_runs["test"], test, epa, bundle)
        reg_output = paths.run_dir / "regression_preds_test_lgbm_regressor.parquet"
        write_parquet(reg_preds, reg_output)
        eval_frame = reg_preds.dropna(subset=["y_true", "pred_factor_value"])
        if not eval_frame.empty:
            dump_json(
                compute_regression_metrics(eval_frame["y_true"].tolist(), eval_frame["pred_factor_value"].tolist()),
                paths.metrics_dir / "regression_metrics_test_lgbm_regressor.json",
            )
            if bool(config.get("whether_uncertainty", True)):
                uncertainty_input = reg_preds.drop(columns=["lower", "upper"], errors="ignore").rename(
                    columns={"lower_conformal": "lower", "upper_conformal": "upper"}
                )
                dump_json(
                    evaluate_uncertainty(uncertainty_input),
                    paths.metrics_dir / "uncertainty_metrics_test_lgbm_regressor.json",
                )

    if bool(config.get("whether_process_extension", False)) and paths.uslci_path is not None:
        uslci = load_uslci_processes(paths.uslci_path)
        process_records = retrieve_process_candidates(
            products_frame=test,
            uslci_frame=uslci,
            retriever_ckpt=str(config.get("process_retriever", "bm25")),
            top_k=int(config.get("process_top_k", 10)),
            prefilter_by_naics=bool(config.get("process_prefilter_by_naics", False)),
            batch_size=batch_size,
        )
        write_jsonl(process_records, paths.run_dir / "process_topk_test.jsonl")

    return manifest
