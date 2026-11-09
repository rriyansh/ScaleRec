
import torch

from src.models.two_tower import TwoTower
from src.models.ranker import Ranker


def test_two_tower_forward():
    model = TwoTower(user_dim=4, item_dim=4, emb_dim=8, hidden_dim=16)
    uf = torch.randn(5, 4)
    itf = torch.randn(5, 4)
    ue, ie = model(uf, itf)
    assert ue.shape == (5, 8)
    assert ie.shape == (5, 8)
    assert torch.allclose(ue.norm(dim=1), torch.ones(5), atol=1e-5)

    loss = model.compute_loss(uf, itf)
    assert loss.item() > 0


def test_ranker_fit_predict(tmp_path):
    import numpy as np
    ranker = Ranker(params={"n_estimators": 5, "max_depth": 2,
                            "objective": "binary:logistic"})
    X = np.random.rand(50, 4)
    y = (X[:, 0] > 0.5).astype(int)
    ranker.fit(X, y)
    preds = ranker.predict_proba(X)
    assert preds.shape == (50,)
    path = tmp_path / "model.json"
    ranker.save(path)
    ranker2 = Ranker(params={})
    ranker2.load(path)
    assert ranker2.predict_proba(X).shape == (50,)
