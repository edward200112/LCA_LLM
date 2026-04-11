from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/multilingual-e5-base",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download open-source retriever or reranker checkpoints for local/offline experiments."
    )
    parser.add_argument(
        "--model",
        choices=["all", *DEFAULT_MODELS.keys()],
        default="all",
        help="Model alias to download.",
    )
    parser.add_argument(
        "--output_dir",
        default="models/retriever/checkpoints",
        help="Directory to store downloaded checkpoints.",
    )
    parser.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Optional allow-list for snapshot files.",
    )
    return parser


def download_one(repo_id: str, output_dir: Path, allow_patterns: list[str] | None = None) -> Path:
    target_dir = output_dir / repo_id.split("/")[-1]
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    return target_dir


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = DEFAULT_MODELS.items() if args.model == "all" else [(args.model, DEFAULT_MODELS[args.model])]
    for alias, repo_id in selected:
        target_dir = download_one(repo_id, output_dir, allow_patterns=args.allow_patterns)
        print(f"{alias}: {repo_id} -> {target_dir}")


if __name__ == "__main__":
    main()
