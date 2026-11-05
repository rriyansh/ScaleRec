import os
import torch
import numpy as np
import faiss
import pandas as pd
from two_tower import TwoTower

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EMBEDDING_DIM = 64
BATCH_SIZE = 1024


def build_index():
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    num_items = len(items)

    model = TwoTower(num_users=10000, num_items=num_items, embedding_dim=EMBEDDING_DIM)
    ckpt = os.path.join(MODEL_DIR, "two_tower.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    item_ids = torch.LongTensor(np.arange(num_items))
    with torch.no_grad():
        vecs = model.item_vector(item_ids).numpy().astype("float32")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vecs)

    os.makedirs(MODEL_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(MODEL_DIR, "faiss_index.bin"))
    print(f"FAISS index saved with {index.ntotal} items")


if __name__ == "__main__":
    build_index()
