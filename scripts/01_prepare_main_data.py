from __future__ import annotations

import argparse
from pathlib import Path

from open_match_lca.constants import LOGS_DIR
from open_match_lca.data.build_naics_corpus import build_naics_corpus
from open_match_lca.data.merge_targets import merge_products_with_epa_factors
from open_match_lca.data.parse_amazon_caml import parse_amazon_caml
from open_match_lca.data.parse_epa_factors import parse_epa_factors
from open_match_lca.data.validate_dataset import write_summary_report
from open_match_lca.io_utils import load_yaml, write_parquet
from open_match_lca.logging_utils import log_final_metrics, setup_run_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon_dir", required=True)
    parser.add_argument("--epa_dir", required=True)
    parser.add_argument("--naics_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    logger, _ = setup_run_logger(
        "01_prepare_main_data",
        LOGS_DIR,
        config_path=args.config,
        dataset_version=config.get("dataset_version"),
    )

    products = parse_amazon_caml(args.amazon_dir)
    epa_factors = parse_epa_factors(args.epa_dir)
    naics = build_naics_corpus(args.naics_dir)

    title_map = naics.set_index("naics_code")["naics_title"].to_dict()
    products["gold_naics_title"] = products["gold_naics_title"].mask(
        products["gold_naics_title"].eq(""),
        products["gold_naics_code"].map(title_map).fillna(""),
    )

    merged = merge_products_with_epa_factors(products, epa_factors)
    out_dir = Path(args.out_dir)
    write_parquet(products, out_dir / "products.parquet")
    write_parquet(epa_factors, out_dir / "epa_factors.parquet")
    write_parquet(naics, out_dir / "naics_corpus.parquet")
    write_parquet(merged, out_dir / "products_with_targets.parquet")
    summary_path = Path(config.get("summary_output", out_dir / "data_summary.json"))
    summary = write_summary_report(products, str(summary_path))
    logger.info("prepared_main_data", extra={"structured": {"summary": summary}})
    log_final_metrics(logger, {"products": len(products), "naics": len(naics), "epa": len(epa_factors)})


if __name__ == "__main__":
    main()
