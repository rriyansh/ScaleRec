import pytest

from src.config import ConfigError, load_config


def test_load_config_defaults(cfg_path):
    cfg = load_config(cfg_path)
    assert cfg.model.embedding_dim > 0
    assert cfg.data.num_users > 0
    assert cfg.ranking.top_k <= cfg.ranking.n_candidates


def test_config_redacted_output(cfg_path):
    cfg = load_config(cfg_path)
    d = cfg.to_dict()
    assert d["security"]["api_key"] == "***REDACTED***"


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        load_config("does/not/exist.yaml")
