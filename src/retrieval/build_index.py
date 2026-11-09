"""Build FAISS index from trained item tower embeddings."""
from pathlib import Path

import faiss
import numpy as np
import torch

from src.config import load_config
from src.logger import get_logger, setup_logging
from src.models.two_tower import TwoTower

logger = get_logger(__name__)


def build_index(config_path: str | None = None) -> None:
    setup_logging()
    cfg = load_config(config_path)
    device = torch.device("cpu")

    proc_dir = cfg.data.processed_dir
    user_feats = torch.tensor(np.load(f"{proc_dir}/user_features.npy"))
    item_feats = torch.tensor(np.load(f"{proc_dir}/item_features.npy"))

    model = TwoTower(
        user_dim=user_feats.shape[1],
        item_dim=item_feats.shape[1],
        emb_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        temperature=cfg.model.temperature,
    )
    model.load_state_dict(torch.load("models_store/two_tower.pt", map_location=device))
    model.eval()

    item_emb = model.encode_items(item_feats).numpy().astype("float32")
    dim = item_emb.shape[1]

    logger.info("Building FAISS index: %d items, dim=%d", item_emb.shape[0], dim)

    if cfg.faiss.index_type == "IVFPQ":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer, dim, cfg.faiss.nlist, cfg.faiss.m, cfg.faiss.nbits
        )
        index.train(item_emb)
        index.add(item_emb)
        index.nprobe = cfg.faiss.nprobe
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(item_emb)

    out_dir = Path("models_store")
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss_index.bin"))
    np.save(out_dir / "item_embeddings.npy", item_emb)
    logger.info("Saved FAISS index (%d vectors)", index.ntotal)


if __name__ == "__main__":
    build_index()
