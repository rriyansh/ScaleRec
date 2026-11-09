"""Train XGBoost ranker on FAISS-retrieved candidates."""
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.logger import get_logger, setup_logging
from src.models.ranker import Ranker
from src.models.two_tower import TwoTower
from src.ranking.features import build_features

logger = get_logger(__name__)


def train(config_path: str | None = None) -> None:
    setup_logging()
    cfg = load_config(config_path)
    proc_dir = cfg.data.processed_dir

    user_feats = torch.tensor(np.load(f"{proc_dir}/user_features.npy"))
    item_feats = torch.tensor(np.load(f"{proc_dir}/item_features.npy"))
    train_df = pd.read_parquet(f"{proc_dir}/train.parquet")

    tower = TwoTower(
        user_dim=user_feats.shape[1],
        item_dim=item_feats.shape[1],
        emb_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        temperature=cfg.model.temperature,
    )
    tower.load_state_dict(torch.load("models_store/two_tower.pt", map_location="cpu"))
    tower.eval()

    user_emb_all = tower.encode_users(user_feats).numpy()
    item_emb_all = tower.encode_items(item_feats).numpy()
    index = faiss.read_index("models_store/faiss_index.bin")

    K = cfg.ranking.n_candidates

    X_list, y_list = [], []
    sample = train_df.sample(min(3000, len(train_df)), random_state=42)

    for _, row in sample.iterrows():
        u = int(row["user_id"])
        i = int(row["item_id"])
        pos = user_emb_all[u : u + 1].astype("float32")
        _, candidates = index.search(pos, K)
        cand_ids = candidates[0]
        feats = build_features(user_emb_all[u], item_emb_all[cand_ids])
        labels = (cand_ids == i).astype(int)
        X_list.append(feats)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    logger.info("Ranker training matrix: %s, positives=%d", X.shape, int(y.sum()))

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ranker = Ranker(params=dict(cfg.ranking.xgb_params.to_dict()))
    ranker.fit(X_tr, y_tr, X_va, y_va)

    auc = roc_auc_score(y_va, ranker.predict_proba(X_va))
    logger.info("Validation AUC: %.4f", auc)

    Path("models_store").mkdir(parents=True, exist_ok=True)
    ranker.save("models_store/xgb_ranker.json")
    logger.info("Saved ranker to models_store/xgb_ranker.json")


if __name__ == "__main__":
    train()
