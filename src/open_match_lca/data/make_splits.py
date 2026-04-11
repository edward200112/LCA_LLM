from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from open_match_lca.io_utils import ensure_directory, write_parquet


def _split_train_dev_test(
    frame: pd.DataFrame,
    stratify: pd.Series | None,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if len(frame) < 3:
        raise RuntimeError(
            f"Need at least 3 rows to create train/dev/test splits, got {len(frame)}"
        )
    if stratify is not None:
        value_counts = stratify.value_counts()
        if frame.shape[0] < 5 or value_counts.min() < 2:
            stratify = None
    train_df, temp_df = train_test_split(
        frame,
        test_size=0.4,
        random_state=seed,
        stratify=stratify if stratify is not None else None,
    )
    temp_stratify = temp_df["gold_naics_code"] if stratify is not None else None
    if temp_stratify is not None and temp_stratify.value_counts().min() < 2:
        temp_stratify = None
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_stratify if temp_stratify is not None else None,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "dev": dev_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def make_dataset_splits(
    products: pd.DataFrame,
    split_type: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if split_type == "random_stratified":
        return _split_train_dev_test(products, products["gold_naics_code"], seed)

    if split_type == "hierarchical_zero_shot":
        code4 = products["gold_naics_code"].str[:4]
        held_out = set(sorted(code4.unique())[::3])
        test_mask = code4.isin(held_out)
        remaining = products.loc[~test_mask].reset_index(drop=True)
        if len(remaining) < 3 or int(test_mask.sum()) == 0:
            return _split_train_dev_test(products, products["gold_naics_code"], seed)
        split = _split_train_dev_test(remaining, remaining["gold_naics_code"], seed)
        split["test"] = products.loc[test_mask].reset_index(drop=True)
        return split

    if split_type == "cluster_ood":
        vectorizer = TfidfVectorizer(min_df=1)
        matrix = vectorizer.fit_transform(products["text"].astype(str))
        cluster_count = min(5, max(2, len(products) // 3))
        labels = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit_predict(matrix)
        products = products.copy()
        products["cluster_id"] = labels
        held_out_cluster = int(sorted(set(labels))[-1])
        test_mask = products["cluster_id"] == held_out_cluster
        remaining = products.loc[~test_mask].drop(columns=["cluster_id"]).reset_index(drop=True)
        if len(remaining) < 3 or int(test_mask.sum()) == 0:
            return _split_train_dev_test(
                products.drop(columns=["cluster_id"]),
                products["gold_naics_code"],
                seed,
            )
        split = _split_train_dev_test(remaining, remaining["gold_naics_code"], seed)
        split["test"] = products.loc[test_mask].drop(columns=["cluster_id"]).reset_index(drop=True)
        return split

    raise RuntimeError(f"Unsupported split_type: {split_type}")


def write_splits(split_map: dict[str, pd.DataFrame], out_dir: str | Path, split_type: str) -> None:
    out_root = Path(out_dir)
    ensure_directory(out_root)
    for split_name, frame in split_map.items():
        write_parquet(frame, out_root / f"{split_type}_{split_name}.parquet")
