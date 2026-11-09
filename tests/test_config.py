from src.config import load_config


def test_load_config(cfg_path):
    cfg = load_config(cfg_path)
    assert cfg.model.embedding_dim == 8
    assert cfg.data.num_users == 50
    assert cfg.ranking.top_k == 5
