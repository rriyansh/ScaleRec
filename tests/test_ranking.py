import numpy as np
import pandas as pd

from src.config import load_config
from src.data.synthetic import generate
from src.data.preprocess import preprocess


def test_generate_and_preprocess(cfg_path):
    generate(cfg_path)
    cfg = load_config(cfg_path)
    raw = cfg.data.raw_dir
    proc = cfg.data.processed_dir

    assert (pd.read_csv(f"{raw}/interactions.csv").shape[0]
            == cfg.data.num_interactions)

    preprocess(cfg_path)
    uf = np.load(f"{proc}/user_features.npy")
    assert uf.shape == (cfg.data.num_users, cfg.data.user_feat_dim)
    assert abs(float(uf.mean())) < 0.1
