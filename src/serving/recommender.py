"""Full recommendation pipeline: user -> retrieval -> ranking."""
from pathlib import Path

import faiss
import numpy as np
import torch

from src.config import Config
from src.logger import get_logger
from src.models.ranker import Ranker
from src.models.two_tower import TwoTower
from src.ranking.features import build_features

logger = get_logger(__name__)


class Recommender:
    def __init__(self, cfg: Config, models_dir: str = "models_store"):
        self.cfg = cfg
        proc_dir = cfg.data.processed_dir
        models_dir = Path(models_dir)

        self.user_feats = torch.tensor(np.load(f"{proc_dir}/user_features.npy"))
        self.item_feats = torch.tensor(np.load(f"{proc_dir}/item_features.npy"))

        self.tower = TwoTower(
            user_dim=self.user_feats.shape[1],
            item_dim=self.item_feats.shape[1],
            emb_dim=cfg.model.embedding_dim,
            hidden_dim=cfg.model.hidden_dim,
            temperature=cfg.model.temperature,
        )
        self.tower.load_state_dict(
            torch.load(models_dir / "two_tower.pt", map_location="cpu")
        )
        self.tower.eval()

        self.item_embeddings = np.load(models_dir / "item_embeddings.npy")
        self.faiss_index = faiss.read_index(str(models_dir / "faiss_index.bin"))

        self.ranker = Ranker(params=dict(cfg.ranking.xgb_params.to_dict()))
        self.ranker.load(models_dir / "xgb_ranker.json")

        self.n_candidates = cfg.ranking.n_candidates
        self.top_k = cfg.ranking.top_k
        logger.info("Recommender initialized: %d items indexed",
                    self.faiss_index.ntotal)

    def recommend(self, user_id: int) -> list[dict]:
        if user_id < 0 or user_id >= self.user_feats.shape[0]:
            raise ValueError(f"user_id {user_id} out of range")

        u_feats = self.user_feats[user_id : user_id + 1]
        with torch.no_grad():
            user_emb = self.tower.encode_users(u_feats).numpy().astype("float32")

        _, candidate_ids = self.faiss_index.search(user_emb, self.n_candidates)
        candidate_ids = candidate_ids[0]

        cand_embs = self.item_embeddings[candidate_ids]
        X = build_features(user_emb[0], cand_embs)
        scores = self.ranker.predict_proba(X)

        top_idx = np.argsort(-scores)[: self.top_k]
        return [
            {"item_id": int(candidate_ids[i]), "score": float(scores[i])}
            for i in top_idx
        ]
