from __future__ import annotations

import argparse
import os
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import dump_json, ensure_directory, load_yaml, write_parquet
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger
from open_match_lca.retrieval.rerank_cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    EpochMetricsCallback,
    load_pair_frame,
    scored_pairs_to_retrieval_records,
    validate_reranker_pair_frame,
    _build_trainer_dataset,
)
from open_match_lca.seed import seed_everything
from open_match_lca.torch_utils import resolve_torch_device

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--dev_pairs", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(2, min(8, cpu_count // 2))


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def _resolve_precision_flags(config: dict, resolved_device: str) -> tuple[bool, bool, bool]:
    if torch is None or not _is_cuda_device(resolved_device):
        return False, False, False
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    bf16 = bool(config.get("bf16", bf16_supported))
    fp16 = bool(config.get("fp16", not bf16))
    tf32 = bool(config.get("tf32", True))
    return fp16, bf16, tf32


def _resolve_train_batch_size(config: dict, resolved_device: str) -> int:
    base_batch_size = int(config.get("batch_size", 8))
    if "train_batch_size" in config:
        return int(config["train_batch_size"])
    if not _is_cuda_device(resolved_device):
        return base_batch_size
    return max(base_batch_size, min(int(config.get("max_train_batch_size", 128)), base_batch_size * 8))


def _resolve_eval_batch_size(config: dict, train_batch_size: int, resolved_device: str) -> int:
    if "eval_batch_size" in config:
        return int(config["eval_batch_size"])
    if _is_cuda_device(resolved_device):
        return max(train_batch_size, min(int(config.get("max_eval_batch_size", 256)), train_batch_size * 2))
    return train_batch_size


def _enable_cuda_speedups(resolved_device: str, tf32: bool) -> None:
    if torch is None or not _is_cuda_device(resolved_device):
        return
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def train_cross_encoder_reranker_optimized(
    *,
    train_pairs,
    dev_pairs,
    base_model: str,
    output_dir: str,
    config: dict,
    logger,
):
    if CrossEncoder is None or CrossEncoderTrainer is None or CrossEncoderTrainingArguments is None:
        raise RuntimeError(
            "sentence-transformers/transformers are not fully installed. Install full project dependencies "
            "before training a reranker."
        )

    validate_reranker_pair_frame(train_pairs, "train_pairs")
    validate_reranker_pair_frame(dev_pairs, "dev_pairs")

    output_root = Path(output_dir)
    ensure_directory(output_root)
    model_dir = output_root / "reranker_model"
    checkpoint_dir = output_root / "checkpoints"
    ensure_directory(checkpoint_dir)

    resolved_device = resolve_torch_device(config.get("device"))
    train_batch_size = _resolve_train_batch_size(config, resolved_device)
    eval_batch_size = _resolve_eval_batch_size(config, train_batch_size, resolved_device)
    fp16, bf16, tf32 = _resolve_precision_flags(config, resolved_device)
    dataloader_num_workers = int(config.get("dataloader_num_workers", _default_num_workers()))
    dataloader_num_workers = max(0, dataloader_num_workers)
    dataloader_prefetch_factor = None if dataloader_num_workers == 0 else int(config.get("dataloader_prefetch_factor", 4))
    auto_find_batch_size = bool(config.get("auto_find_batch_size", _is_cuda_device(resolved_device)))
    epochs = int(config.get("epochs", 1))
    learning_rate = float(config.get("learning_rate", 2e-5))
    max_length = int(config.get("max_length", 256))
    top_k = int(config.get("top_k", 10))
    checkpoint_save_steps = int(config.get("checkpoint_save_steps", 300))
    checkpoint_save_total_limit = int(config.get("checkpoint_save_total_limit", 2))

    _enable_cuda_speedups(resolved_device, tf32)

    model = CrossEncoder(base_model, num_labels=1, max_length=max_length, device=resolved_device)
    train_dataset = _build_trainer_dataset(train_pairs)
    steps_per_epoch = max(1, (len(train_pairs) + train_batch_size - 1) // train_batch_size)
    warmup_steps = max(0, int(steps_per_epoch * epochs * 0.1))
    epoch_metrics_history: list[dict[str, float | int]] = []

    if logger is not None:
        logger.info(
            "reranker_optimized_training_config",
            extra={
                "structured": {
                    "device": resolved_device,
                    "train_batch_size": train_batch_size,
                    "eval_batch_size": eval_batch_size,
                    "fp16": fp16,
                    "bf16": bf16,
                    "tf32": tf32,
                    "auto_find_batch_size": auto_find_batch_size,
                    "dataloader_num_workers": dataloader_num_workers,
                    "dataloader_prefetch_factor": dataloader_prefetch_factor,
                }
            },
        )

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
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
        fp16=fp16,
        bf16=bf16,
        tf32=tf32,
        auto_find_batch_size=auto_find_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=bool(config.get("dataloader_pin_memory", True)),
        dataloader_persistent_workers=bool(
            config.get("dataloader_persistent_workers", dataloader_num_workers > 0)
        ),
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        report_to=[],
    )
    callbacks = [
        EpochMetricsCallback(
            cross_encoder=model,
            dev_pairs=dev_pairs,
            epoch_metrics_history=epoch_metrics_history,
            top_k=top_k,
            batch_size=eval_batch_size,
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

    effective_train_batch_size = int(getattr(trainer, "_train_batch_size", train_batch_size))
    dev_scored = dev_pairs.copy()
    dev_inputs = dev_scored[["query_text", "candidate_text"]].astype(str).values.tolist()
    dev_scored["rerank_score"] = model.predict(
        dev_inputs,
        batch_size=eval_batch_size,
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
            "requested_train_batch_size": train_batch_size,
            "effective_train_batch_size": effective_train_batch_size,
            "eval_batch_size": eval_batch_size,
            "fp16": fp16,
            "bf16": bf16,
            "tf32": tf32,
            "auto_find_batch_size": auto_find_batch_size,
            "dataloader_num_workers": dataloader_num_workers,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_save_steps": checkpoint_save_steps,
            "checkpoint_save_total_limit": checkpoint_save_total_limit,
        },
        output_root / "reranker_training_summary.json",
    )
    dump_json(epoch_metrics_history, output_root / "reranker_epoch_metrics.json")

    if logger is not None:
        dev_records = scored_pairs_to_retrieval_records(dev_scored, top_k=top_k)
        logger.info(
            "reranker_optimized_training_completed",
            extra={
                "structured": {
                    "dev_query_count": len(dev_records),
                    "effective_train_batch_size": effective_train_batch_size,
                }
            },
        )

    return {
        "model_dir": model_dir,
        "checkpoint_dir": checkpoint_dir,
        "dev_pair_scores_path": dev_pair_scores_path,
        "train_pair_count": int(len(train_pairs)),
        "dev_pair_count": int(len(dev_pairs)),
    }


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    config = load_yaml(args.config)
    logger, _ = setup_run_logger("06_train_reranker", LOGS_DIR, config_path=args.config, seed=args.seed)
    train_pairs = load_pair_frame(args.train_pairs)
    dev_pairs = load_pair_frame(args.dev_pairs)
    artifacts = train_cross_encoder_reranker_optimized(
        train_pairs=train_pairs,
        dev_pairs=dev_pairs,
        base_model=args.base_model,
        output_dir=args.output_dir,
        config=config,
        logger=logger,
    )
    log_final_metrics(
        logger,
        {
            "model_dir": str(artifacts["model_dir"]),
            "checkpoint_dir": str(artifacts["checkpoint_dir"]),
            "dev_pair_scores_path": str(artifacts["dev_pair_scores_path"]),
            "train_pair_count": artifacts["train_pair_count"],
            "dev_pair_count": artifacts["dev_pair_count"],
        },
    )


if __name__ == "__main__":
    main()
