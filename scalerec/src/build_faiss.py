import numpy as np
import torch
import faiss
from two_tower import TwoTower

user_feats = torch.tensor(np.load("data/user_features.npy"))
item_feats = torch.tensor(np.load("data/item_features.npy"))

model = TwoTower(user_feats.shape[1], item_feats.shape[1], emb_dim=32)
model.load_state_dict(torch.load("models/two_tower.pt"))
model.eval()

# ---- Saare item embeddings nikaalo ----
with torch.no_grad():
    _, item_emb = model(user_feats[:1], item_feats)  # dummy user
item_emb = item_emb.numpy().astype("float32")

# ---- FAISS index (Inner Product) ----
dim = item_emb.shape[1]
index = faiss.IndexFlatIP(dim)   # basic; production mein IVFPQ use karo
index.add(item_emb)

faiss.write_index(index, "models/faiss_index.bin")
print(f"✅ FAISS index built with {index.ntotal} items")
