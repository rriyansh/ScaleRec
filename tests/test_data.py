import numpy as np
import pandas as pd

from src.config import load_config
from src.data.synthetic import generate
from src.data.preprocess import preprocess
from src.data.validation import validate_interactions, validate_features
from src.exceptions import ValidationError


def test_generate_and_preprocess(cfg_path):
    generate(cfg_path)
    cfg = load_config(cfg_path)
    raw = cfg.data.raw_dir
    proc = cfg.data.processed_dir

    df = pd.read_csv(f"{raw}/interactions.csv")
    assert len(df) == cfg.data.num_interactions

    preprocess(cfg_path)
    uf = np.load(f"{proc}/user_features.npy")
    assert uf.shape == (cfg.data.num_users, cfg.data.user_feat_dim)
    assert abs(float(uf.mean())) < 0.2


def test_validate_features_rejects_nan():
    arr = np.array([[1.0, np.nan]], dtype="float32")
    with pytest.raises(ValidationError):
        validate_features(arr, 1, 2, "test")


def test_validate_interactions_rejects_negative():
    df = pd.DataFrame({"user_id": [-1], "item_id": [0], "clicked": [1]})
    with pytest.raises(ValidationError):
        validate_interactions(df, 10, 10, "test")
