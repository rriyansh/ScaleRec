from src.config import load_config
from src.data.synthetic import generate
from src.data.preprocess import preprocess
from src.retrieval.train import train as train_retrieval
from src.retrieval.build_index import build_index
from src.ranking.train import train as train_ranking


def test_ranking_pipeline(cfg_path):
    generate(cfg_path)
    preprocess(cfg_path)
    train_retrieval(cfg_path)
    build_index(cfg_path)
    train_ranking(cfg_path)

    import os
    assert os.path.exists("models_store/xgb_ranker.json")
