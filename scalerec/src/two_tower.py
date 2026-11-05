import torch
import torch.nn as nn


class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=(128, 64)):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        def tower(in_dim):
            layers = []
            prev = in_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
                prev = h
            layers.append(nn.Linear(prev, embedding_dim))
            return nn.Sequential(*layers)

        self.user_tower = tower(embedding_dim)
        self.item_tower = tower(embedding_dim)

    def forward(self, user_ids, item_ids):
        u = self.user_tower(self.user_embedding(user_ids))
        i = self.item_tower(self.item_embedding(item_ids))
        u = nn.functional.normalize(u, p=2, dim=1)
        i = nn.functional.normalize(i, p=2, dim=1)
        return (u * i).sum(dim=1)

    def user_vector(self, user_ids):
        return nn.functional.normalize(self.user_tower(self.user_embedding(user_ids)), p=2, dim=1)

    def item_vector(self, item_ids):
        return nn.functional.normalize(self.item_tower(self.item_embedding(item_ids)), p=2, dim=1)
