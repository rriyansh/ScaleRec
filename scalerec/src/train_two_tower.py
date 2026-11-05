import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from two_tower import TwoTower

# ---- Load data ----
user_feats = torch.tensor(np.load("data/user_features.npy"))
item_feats = torch.tensor(np.load("data/item_features.npy"))
inter = pd.read_csv("data/interactions.csv")
pos = inter[inter.clicked == 1].reset_index(drop=True)


class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = df.user_id.values
        self.items = df.item_id.values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]


loader = DataLoader(InteractionDataset(pos), batch_size=256, shuffle=True)

# ---- Model ----
model = TwoTower(user_feats.shape[1], item_feats.shape[1], emb_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- Train ----
EPOCHS = 5
for epoch in range(EPOCHS):
    total_loss = 0
    for u_ids, i_ids in loader:
        u_emb, i_emb = model(user_feats[u_ids], item_feats[i_ids])

        # Similarity matrix: [B, B]
        logits = u_emb @ i_emb.T / 0.07  # temperature scaling
        labels = torch.arange(logits.size(0))  # diagonal = positive
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "models/two_tower.pt")
print("✅ Two-Tower model saved!")
