import faiss

from src.config import load_config


def test_faiss_index_exists(cfg_path, prepare_artifacts):
    idx = faiss.read_index("models_store/faiss_index.bin")
    cfg = load_config(cfg_path)
    assert idx.ntotal == cfg.data.num_items


def test_item_embeddings_match_index(cfg_path, prepare_artifacts):
    import numpy as np
    emb = np.load("models_store/item_embeddings.npy")
    cfg = load_config(cfg_path)
    assert emb.shape[0] == cfg.data.num_items
    assert emb.shape[1] == cfg.model.embedding_dim
