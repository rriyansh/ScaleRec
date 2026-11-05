import os
import json
import faiss
import numpy as np
import torch
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from two_tower import TwoTower

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EMBEDDING_DIM = 64
TOP_K = 100

app = FastAPI(title="ScaleRec Inference API")

users_df = None
items_df = None
model = None
index = None
ranker = None


class RecRequest(BaseModel):
    user_id: int
    top_k: int = 10


class RecItem(BaseModel):
    item_id: int
    score: float


@app.on_event("startup")
def load_artifacts():
    global users_df, items_df, model, index, ranker
    users_df = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    items_df = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))

    num_items = len(items_df)
    model = TwoTower(num_users=len(users_df), num_items=num_items, embedding_dim=EMBEDDING_DIM)
    ckpt = os.path.join(MODEL_DIR, "two_tower.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    index_path = os.path.join(MODEL_DIR, "faiss_index.bin")
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)

    ranker_path = os.path.join(MODEL_DIR, "xgb_ranker.json")
    if os.path.exists(ranker_path):
        ranker = xgb.XGBClassifier()
        ranker.load_model(ranker_path)


@app.get("/")
def root():
    return {"status": "ok", "service": "ScaleRec"}


@app.post("/recommend", response_model=List[RecItem])
def recommend(req: RecRequest):
    if model is None or index is None:
        return []

    with torch.no_grad():
        u_vec = model.user_vector(torch.LongTensor([req.user_id])).numpy().astype("float32")

    _, item_ids = index.search(u_vec, min(TOP_K, index.ntotal))
    candidates = item_ids[0].tolist()

    if ranker is not None and items_df is not None:
        cand_df = items_df[items_df["item_id"].isin(candidates)].copy()
        user_row = users_df[users_df["user_id"] == req.user_id]
        if len(user_row) and len(cand_df):
            u = user_row.iloc[0]
            feat = pd.concat([pd.DataFrame([u.to_dict()])] * len(cand_df), ignore_index=True)
            feat = pd.concat([feat.reset_index(drop=True), cand_df.reset_index(drop=True)], axis=1)
            feat["price_log"] = np.log1p(feat["price"])
            cat_cols = ["gender", "city", "category", "brand"]
            feat = pd.get_dummies(feat, columns=cat_cols, drop_first=True)
            model_feats = ranker.get_booster().feature_names
            for col in model_feats:
                if col not in feat.columns:
                    feat[col] = 0
            feat = feat[model_feats]
            scores = ranker.predict_proba(feat)[:, 1]
            cand_df["score"] = scores
            cand_df = cand_df.sort_values("score", ascending=False).head(req.top_k)
        else:
            cand_df = cand_df.head(req.top_k)
            cand_df["score"] = 0.5
    else:
        cand_df = items_df[items_df["item_id"].isin(candidates)].head(req.top_k).copy()
        cand_df["score"] = 0.5

    return [RecItem(item_id=int(r.item_id), score=float(r.score)) for r in cand_df.itertuples()]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
