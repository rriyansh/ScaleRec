"""Preprocessing: normalize features and persist to processed dir."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.logger import get_logger, setup_logging
from src.utils.io import ensure_dir, save_npy

logger = get_logger(__name__)


def preprocess(config_path: str | None = None) -> None:
    setup_logging()
    cfg = load_config(config_path)
    data_cfg = cfg.data

    raw_dir = data_cfg.raw_dir
    proc_dir = ensure_dir(data_cfg.processed_dir)

    logger.info("Loading raw data from %s", raw_dir)
    user_feats = np.load(f"{raw_dir}/user_features.npy")
    item_feats = np.load(f"{raw_dir}/item_features.npy")
    train_df = pd.read_csv(f"{raw_dir}/train.csv")
    test_df = pd.read_csv(f"{raw_dir}/test.csv")

    # Normalize features
    user_scaler = StandardScaler()
    item_scaler = StandardScaler()
    user_feats = user_scaler.fit_transform(user_feats).astype("float32")
    item_feats = item_scaler.fit_transform(item_feats).astype("float32")

    save_npy(user_feats, f"{proc_dir}/user_features.npy")
    save_npy(item_feats, f"{proc_dir}/item_features.npy")
    train_df.to_parquet(f"{proc_dir}/train.parquet", index=False)
    test_df.to_parquet(f"{proc_dir}/test.parquet", index=False)

    logger.info("Preprocessing complete. Saved to %s", proc_dir)


if __name__ == "__main__":
    preprocess()
