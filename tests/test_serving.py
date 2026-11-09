from fastapi.testclient import TestClient

from src.config import load_config
from src.data.synthetic import generate
from src.data.preprocess import preprocess
from src.retrieval.train import train as train_retrieval
from src.retrieval.build_index import build_index
from src.ranking.train import train as train_ranking
from src.serving.api import app


def test_api_endpoint(cfg_path):
    generate(cfg_path)
    preprocess(cfg_path)
    train_retrieval(cfg_path)
    build_index(cfg_path)
    train_ranking(cfg_path)

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

        r = client.get("/recommend/0")
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == 0
        assert len(data["recommendations"]) <= 5

        r = client.get("/recommend/99999")
        assert r.status_code == 400
