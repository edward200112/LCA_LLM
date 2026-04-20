from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from open_match_lca.data.aux_corpus import extract_aux_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Attempt OCR via open_match_lca.data.ocr_adapter after non-OCR parsing fails.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extract_aux_documents(args.repo_root.resolve(), enable_ocr=args.enable_ocr)


if __name__ == "__main__":
    main()
