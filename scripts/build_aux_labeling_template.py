from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.data.aux_corpus import build_labeling_template


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
    build_labeling_template(args.repo_root.resolve())


if __name__ == "__main__":
    main()
