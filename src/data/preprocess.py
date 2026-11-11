"""Preprocess raw data: normalize features, validate schema, save to processed dir."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data.validation import validate_features, validate_interactions
from src.logger import get_logger, setup_logging
from src.utils.io import ensure_dir, save_json, save_npy

logger = get_logger(__name__)


def preprocess(config_path: str | None = None) -> dict:
    setup_logging()
    cfg = load_config(config_path)
    d = cfg.data

    raw_dir = d.raw_dir
    proc_dir = ensure_dir(d.processed_dir)

    logger.info("Loading raw data from %s", raw_dir)
    user_feats = np.load(f"{raw_dir}/user_features.npy")
    item_feats = np.load(f"{raw_dir}/item_features.npy")
    train_df = pd.read_csv(f"{raw_dir}/train.csv")
    test_df = pd.read_csv(f"{raw_dir}/test.csv")

    validate_features(user_feats, d.num_users, d.user_feat_dim, "user")
    validate_features(item_feats, d.num_items, d.item_feat_dim, "item")
    validate_interactions(train_df, d.num_users, d.num_items, "train")
    validate_interactions(test_df, d.num_users, d.num_items, "test")

    user_scaler = StandardScaler()
    item_scaler = StandardScaler()
    user_feats = user_scaler.fit_transform(user_feats).astype("float32")
    item_feats = item_scaler.fit_transform(item_feats).astype("float32")

    save_npy(user_feats, f"{proc_dir}/user_features.npy")
    save_npy(item_feats, f"{proc_dir}/item_features.npy")
    train_df.to_parquet(f"{proc_dir}/train.parquet", index=False)
    test_df.to_parquet(f"{proc_dir}/test.parquet", index=False)

    meta = {
        "num_users": int(user_feats.shape[0]),
        "num_items": int(item_feats.shape[0]),
        "user_feat_dim": int(user_feats.shape[1]),
        "item_feat_dim": int(item_feats.shape[1]),
        "num_train": int(len(train_df)),
        "num_test": int(len(test_df)),
    }
    save_json(meta, f"{proc_dir}/meta.json")

    logger.info("Preprocessing complete: %s", meta)
    return meta


if __name__ == "__main__":
    preprocess()
