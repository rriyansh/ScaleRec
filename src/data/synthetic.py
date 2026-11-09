"""Generate synthetic interaction data for local development."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.logger import get_logger, setup_logging
from src.utils.io import ensure_dir

logger = get_logger(__name__)


def generate(config_path: str | None = None) -> None:
    setup_logging()
    cfg = load_config(config_path)
    data_cfg = cfg.data

    seed = data_cfg.seed
    rng = np.random.default_rng(seed)

    raw_dir = ensure_dir(data_cfg.raw_dir)
    num_users = data_cfg.num_users
    num_items = data_cfg.num_items
    num_interactions = data_cfg.num_interactions
    user_feat_dim = data_cfg.user_feat_dim
    item_feat_dim = data_cfg.item_feat_dim

    logger.info(
        "Generating synthetic data: users=%d, items=%d, interactions=%d",
        num_users, num_items, num_interactions,
    )

    user_features = rng.random((num_users, user_feat_dim)).astype("float32")
    item_features = rng.random((num_items, item_feat_dim)).astype("float32")

    user_ids = rng.integers(0, num_users, num_interactions)
    item_ids = rng.integers(0, num_items, num_interactions)

    # Simulate CTR based on embedding similarity (a learnable signal)
    u = user_features[user_ids]
    i = item_features[item_ids]
    affinity = (u * i).sum(axis=1)
    affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min() + 1e-9)
    clicked = (rng.random(num_interactions) < affinity).astype(int)

    interactions = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "clicked": clicked,
    })

    np.save(raw_dir / "user_features.npy", user_features)
    np.save(raw_dir / "item_features.npy", item_features)
    interactions.to_csv(raw_dir / "interactions.csv", index=False)

    # Train / test split
    train_df, test_df = train_test_split(
        interactions, test_size=0.2, random_state=seed
    )
    train_df.to_csv(raw_dir / "train.csv", index=False)
    test_df.to_csv(raw_dir / "test.csv", index=False)

    logger.info(
        "Saved: %d interactions (%d positives), %d user features, %d item features",
        len(interactions), int(clicked.sum()), num_users, num_items,
    )


if __name__ == "__main__":
    generate()
