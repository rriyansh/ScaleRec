import numpy as np
import torch
import faiss
import xgboost as xgb
from fastapi import FastAPI
from two_tower import TwoTower

app = FastAPI(title="ScaleRec")

# ---- Load everything once ----
user_feats = torch.tensor(np.load("data/user_features.npy"))
item_feats = torch.tensor(np.load("data/item_features.npy"))

tower = TwoTower(user_feats.shape[1], item_feats.shape[1], emb_dim=32)
tower.load_state_dict(torch.load("models/two_tower.pt"))
tower.eval()

faiss_index = faiss.read_index("models/faiss_index.bin")

ranker = xgb.XGBClassifier()
ranker.load_model("models/xgb_ranker.json")

TOPK_RETRIEVE = 50
TOPK_FINAL = 10


@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    # 1. User embedding
    with torch.no_grad():
        u_emb, _ = tower(user_feats[user_id:user_id+1], item_feats[:1])
    u_emb = u_emb.numpy().astype("float32")

    # 2. FAISS retrieval
    _, candidate_ids = faiss_index.search(u_emb, TOPK_RETRIEVE)
    candidate_ids = candidate_ids[0]

    # 3. Ranker ke liye features
    with torch.no_grad():
        _, all_item_emb = tower(user_feats[:1], item_feats)
    all_item_emb = all_item_emb.numpy()

    u_vec = np.repeat(u_emb, len(candidate_ids), axis=0)
    i_vec = all_item_emb[candidate_ids]
    dot = (u_vec * i_vec).sum(axis=1, keepdims=True)
    X = np.hstack([u_vec, i_vec, dot, u_vec * i_vec])

    # 4. CTR scores + sort
    scores = ranker.predict_proba(X)[:, 1]
    ranked = candidate_ids[np.argsort(-scores)][:TOPK_FINAL]

    return {"user_id": user_id, "recommendations": ranked.tolist()}
