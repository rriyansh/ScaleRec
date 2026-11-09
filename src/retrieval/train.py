"""Train the Two-Tower model on positive interactions."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import InteractionDataset
from src.logger import get_logger, setup_logging
from src.models.two_tower import TwoTower

logger = get_logger(__name__)


def train(config_path: str | None = None) -> None:
    setup_logging()
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    proc_dir = cfg.data.processed_dir
    user_feats = torch.tensor(np.load(f"{proc_dir}/user_features.npy"))
    item_feats = torch.tensor(np.load(f"{proc_dir}/item_features.npy"))
    train_df = pd.read_parquet(f"{proc_dir}/train.parquet")

    dataset = InteractionDataset(train_df)
    loader = DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
    )
    logger.info("Training samples: %d", len(dataset))

    model = TwoTower(
        user_dim=user_feats.shape[1],
        item_dim=item_feats.shape[1],
        emb_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        temperature=cfg.model.temperature,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for u_ids, i_ids in loader:
            u_feats = user_feats[u_ids].to(device)
            i_feats = item_feats[i_ids].to(device)
            loss = model.compute_loss(u_feats, i_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        logger.info("Epoch %d/%d | loss=%.4f",
                    epoch + 1, cfg.training.epochs, total_loss / max(steps, 1))

    out_path = Path("models_store/two_tower.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    logger.info("Saved model to %s", out_path)


if __name__ == "__main__":
    train()
