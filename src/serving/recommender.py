"""Full recommendation pipeline: user embedding -> FAISS -> XGBoost."""
from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np
import torch

from src.config import Config
from src.exceptions import ModelNotReadyError, ValidationError
from src.logger import get_logger
from src.metrics import FAISS_SEARCH_LATENCY, RANKER_LATENCY, RECOMMENDATION_LATENCY
from src.models.ranker import Ranker
from src.models.versioning import verify_version
from src.ranking.features import build_features
from src.retrieval.build_index import load_trained_tower

logger = get_logger(__name__)


class Recommender:
    def __init__(self, cfg: Config, models_dir: str | Path = "models_store"):
        self.cfg = cfg
        self.models_dir = Path(models_dir)

        proc_dir = cfg.data.processed_dir
        try:
            self.user_feats = torch.tensor(np.load(f"{proc_dir}/user_features.npy"))
            self.item_feats = torch.tensor(np.load(f"{proc_dir}/item_features.npy"))
        except FileNotFoundError as e:
            raise ModelNotReadyError(f"Data files missing: {e}") from e

        try:
            self.tower, _ = load_trained_tower(self.models_dir / "two_tower.pt")
        except FileNotFoundError as e:
            raise ModelNotReadyError(f"Two-Tower model missing: {e}") from e

        self.item_embeddings = np.load(self.models_dir / "item_embeddings.npy")
        self.faiss_index = faiss.read_index(str(self.models_dir / "faiss_index.bin"))

        ranker_params = dict(cfg.ranking.xgb_params)
        self.ranker = Ranker(params=ranker_params)
        try:
            self.ranker.load(self.models_dir / "xgb_ranker.json")
        except FileNotFoundError as e:
            raise ModelNotReadyError(f"Ranker model missing: {e}") from e

        self.num_users = self.user_feats.shape[0]
        self.n_candidates = cfg.ranking.n_candidates
        self.top_k = cfg.ranking.top_k

        if verify_version(self.models_dir):
            logger.info("Model artifacts verified (SHA256)")
        else:
            logger.warning("Model version verification failed or missing")

        logger.info(
            "Recommender ready users=%d items=%d",
            self.num_users, self.faiss_index.ntotal,
        )

    def _user_embedding(self, user_id: int) -> np.ndarray:
        if user_id < 0 or user_id >= self.num_users:
            raise ValidationError(
                f"user_id {user_id} out of range [0, {self.num_users})"
            )
        u_feats = self.user_feats[user_id : user_id + 1]
        with torch.no_grad():
            emb = self.tower.encode_users(u_feats).numpy().astype("float32")
        return emb

    def recommend(self, user_id: int) -> list[dict]:
        start = time.perf_counter()
        try:
            user_emb = self._user_embedding(user_id)

            t0 = time.perf_counter()
            _, candidate_ids = self.faiss_index.search(user_emb, self.n_candidates)
            FAISS_SEARCH_LATENCY.observe(time.perf_counter() - t0)
            candidate_ids = candidate_ids[0]

            cand_embs = self.item_embeddings[candidate_ids]
            X = build_features(user_emb[0], cand_embs)

            t0 = time.perf_counter()
            scores = self.ranker.predict_proba(X)
            RANKER_LATENCY.observe(time.perf_counter() - t0)

            top_idx = np.argsort(-scores)[: self.top_k]
            result = [
                {"item_id": int(candidate_ids[i]), "score": float(scores[i])}
                for i in top_idx
            ]
            return result
        finally:
            RECOMMENDATION_LATENCY.observe(time.perf_counter() - start)

    def health_check(self) -> dict[str, bool]:
        return {
            "tower_loaded": self.tower is not None,
            "faiss_loaded": self.faiss_index is not None,
            "ranker_loaded": self.ranker.model is not None,
            "num_items": self.faiss_index.ntotal > 0,
        }
