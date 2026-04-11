from __future__ import annotations

import argparse
from pathlib import Path

from open_match_lca.constants import LOGS_DIR
from open_match_lca.io_utils import require_exists
from open_match_lca.logging_utils import setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products_path", required=True)
    parser.add_argument("--uslci_path", required=True)
    parser.add_argument("--prefilter_by_naics", choices=["true", "false"], required=True)
    parser.add_argument("--retriever_ckpt", required=True)
    parser.add_argument("--reranker_ckpt", required=False)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_run_logger("11_run_process_extension", LOGS_DIR)
    require_exists(Path(args.products_path))
    uslci_path = Path(args.uslci_path)
    if not uslci_path.exists():
        raise FileNotFoundError(
            f"USLCI data not found at {uslci_path}. "
            "This feature is an optional extension; the main experiment is unaffected."
        )
    raise RuntimeError("Process extension retrieval is scaffolded but not implemented yet.")


if __name__ == "__main__":
    main()
