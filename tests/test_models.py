import pytest
import torch

from src.models.ranker import Ranker
from src.models.two_tower import TwoTower


def test_two_tower_forward():
    m = TwoTower(user_dim=4, item_dim=4, emb_dim=8, hidden_dim=16)
    u = torch.randn(5, 4)
    i = torch.randn(5, 4)
    ue, ie = m(u, i)
    assert ue.shape == (5, 8)
    assert ie.shape == (5, 8)
    assert torch.allclose(ue.norm(dim=1), torch.ones(5), atol=1e-5)
    assert m.compute_loss(u, i).item() > 0


def test_two_tower_rejects_invalid_config():
    with pytest.raises(ValueError):
        TwoTower(user_dim=4, item_dim=4, temperature=0)


def test_ranker_roundtrip(tmp_path):
    import numpy as np

    r = Ranker(params={"n_estimators": 5, "max_depth": 2,
                       "objective": "binary:logistic", "eval_metric": "auc"})
    X = np.random.rand(50, 4)
    y = (X[:, 0] > 0.5).astype(int)
    r.fit(X, y)
    p = r.predict_proba(X)
    assert p.shape == (50,)

    path = tmp_path / "ranker.json"
    r.save(path)
    r2 = Ranker(params={})
    r2.load(path)
    assert r2.predict_proba(X).shape == (50,)
