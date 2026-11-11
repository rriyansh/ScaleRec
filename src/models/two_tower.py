"""Two-Tower neural recommender model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """MLP encoder: dense features -> L2-normalized embedding."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or emb_dim <= 0:
            raise ValueError("Tower dims must be positive")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TwoTower(nn.Module):
    """User + item towers with in-batch softmax contrastive loss."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.user_tower = Tower(user_dim, hidden_dim, emb_dim, dropout)
        self.item_tower = Tower(item_dim, hidden_dim, emb_dim, dropout)
        self.temperature = temperature

    def forward(
        self, user_feats: torch.Tensor, item_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_feats), self.item_tower(item_feats)

    def compute_loss(
        self, user_feats: torch.Tensor, item_feats: torch.Tensor
    ) -> torch.Tensor:
        user_emb, item_emb = self.forward(user_feats, item_feats)
        logits = user_emb @ item_emb.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_users(self, user_feats: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.user_tower(user_feats)
        self.train(was_training)
        return out

    @torch.no_grad()
    def encode_items(self, item_feats: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.item_tower(item_feats)
        self.train(was_training)
        return out
