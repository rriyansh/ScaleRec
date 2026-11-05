import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from two_tower import TwoTower
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EPOCHS = 5
BATCH_SIZE = 1024
LR = 1e-3
EMBEDDING_DIM = 64


def load_data():
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    num_users = len(users)
    num_items = len(items)
    return interactions, num_users, num_items


def train():
    interactions, num_users, num_items = load_data()

    user_ids = torch.LongTensor(interactions["user_id"].values)
    item_ids = torch.LongTensor(interactions["item_id"].values)
    labels = torch.FloatTensor(interactions["clicked"].values)

    dataset = TensorDataset(user_ids, item_ids, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TwoTower(num_users, num_items, embedding_dim=EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for u, i, y in loader:
            optimizer.zero_grad()
            logits = model(u, i)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} loss={total_loss/len(loader):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "two_tower.pt"))
    print(f"Saved model to {MODEL_DIR}/two_tower.pt")


if __name__ == "__main__":
    train()
