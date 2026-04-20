from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.data.aux_corpus import build_all_pdf_indexes, normalize_aux_directories, write_data_readme


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    normalize_aux_directories(repo_root)
    build_all_pdf_indexes(repo_root)
    write_data_readme(repo_root)


if __name__ == "__main__":
    main()
