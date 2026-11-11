"""Build FAISS index from trained item tower embeddings."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import torch

from src.config import load_config
from src.logger import get_logger, setup_logging
from src.models.two_tower import TwoTower

logger = get_logger(__name__)


def load_trained_tower(path: str | Path, device: str = "cpu") -> tuple[TwoTower, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TwoTower(
        user_dim=cfg["user_dim"],
        item_dim=cfg["item_dim"],
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        temperature=cfg["temperature"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def build_index(config_path: str | None = None) -> Path:
    setup_logging()
    cfg = load_config(config_path)
    proc_dir = cfg.data.processed_dir

    item_feats = torch.tensor(np.load(f"{proc_dir}/item_features.npy"))
    model, model_cfg = load_trained_tower("models_store/two_tower.pt")

    item_emb = model.encode_items(item_feats).numpy().astype("float32")
    dim = item_emb.shape[1]
    n = item_emb.shape[0]
    logger.info("Building FAISS index items=%d dim=%d type=%s",
                n, dim, cfg.faiss.index_type)

    if cfg.faiss.index_type == "IVFPQ":
        if n < cfg.faiss.nlist:
            raise ValueError(
                f"IVFPQ needs n_items >= nlist ({n} < {cfg.faiss.nlist})"
            )
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

    out = Path("models_store")
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss_index.bin"))
    np.save(out / "item_embeddings.npy", item_emb)
    logger.info("FAISS index written: %d vectors", index.ntotal)
    return out / "faiss_index.bin"


if __name__ == "__main__":
    build_index()
