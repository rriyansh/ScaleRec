import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """Ek simple MLP jo features ko embedding mein convert karta hai."""
    def __init__(self, in_dim, emb_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, emb_dim=32):
        super().__init__()
        self.user_tower = Tower(user_dim, emb_dim)
        self.item_tower = Tower(item_dim, emb_dim)

    def forward(self, user_feat, item_feat):
        return self.user_tower(user_feat), self.item_tower(item_feat)
