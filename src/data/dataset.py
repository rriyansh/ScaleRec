import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Dataset of positive (user, item) interaction pairs."""

    def __init__(self, df: pd.DataFrame):
        positives = df[df["clicked"] == 1].reset_index(drop=True)
        self.user_ids = torch.tensor(positives["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(positives["item_id"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int):
        return self.user_ids[idx], self.item_ids[idx]
