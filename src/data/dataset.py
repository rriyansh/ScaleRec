"""PyTorch datasets."""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Dataset of positive (user, item) interaction pairs."""

    def __init__(self, df: pd.DataFrame, require_positive: bool = True):
        if require_positive:
            df = df[df["clicked"] == 1]
        if df.empty:
            raise ValueError("InteractionDataset: empty after filtering")
        df = df.reset_index(drop=True)
        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(df["item_id"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx]
