"""Generate synthetic interaction data for local development and testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.logger import get_logger, setup_logging
from src.utils.io import ensure_dir, save_npy

logger = get_logger(__name__)


def generate(config_path: str | None = None) -> dict:
    setup_logging()
    cfg = load_config(config_path)
    d = cfg.data

    rng = np.random.default_rng(d.seed)
    raw_dir = ensure_dir(d.raw_dir)

    logger.info(
        "Generating synthetic data users=%d items=%d interactions=%d",
        d.num_users, d.num_items, d.num_interactions,
    )

    user_features = rng.random((d.num_users, d.user_feat_dim)).astype("float32")
    item_features = rng.random((d.num_items, d.item_feat_dim)).astype("float32")

    user_ids = rng.integers(0, d.num_users, d.num_interactions)
    item_ids = rng.integers(0, d.num_items, d.num_interactions)

    u = user_features[user_ids]
    i = item_features[item_ids]
    affinity = (u * i).sum(axis=1)
    lo, hi = affinity.min(), affinity.max()
    affinity = (affinity - lo) / (hi - lo + 1e-9)
    clicked = (rng.random(d.num_interactions) < affinity).astype(int)

    interactions = pd.DataFrame(
        {"user_id": user_ids, "item_id": item_ids, "clicked": clicked}
    )

    save_npy(user_features, raw_dir / "user_features.npy")
    save_npy(item_features, raw_dir / "item_features.npy")
    interactions.to_csv(raw_dir / "interactions.csv", index=False)

    train_df, test_df = train_test_split(
        interactions, test_size=0.2, random_state=d.seed
    )
    train_df.to_csv(raw_dir / "train.csv", index=False)
    test_df.to_csv(raw_dir / "test.csv", index=False)

    stats = {
        "num_interactions": int(len(interactions)),
        "num_positives": int(clicked.sum()),
        "num_users": d.num_users,
        "num_items": d.num_items,
    }
    logger.info("Synthetic data ready: %s", stats)
    return stats


if __name__ == "__main__":
    generate()
