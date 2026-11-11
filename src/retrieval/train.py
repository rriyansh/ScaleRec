"""Train the Two-Tower model on positive interactions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import InteractionDataset
from src.logger import get_logger, setup_logging
from src.models.two_tower import TwoTower
from src.models.versioning import record_version

logger = get_logger(__name__)


def train(config_path: str | None = None) -> Path:
    setup_logging()
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device=%s", device)

    proc_dir = cfg.data.processed_dir
    user_feats = torch.tensor(np.load(f"{proc_dir}/user_features.npy"))
    item_feats = torch.tensor(np.load(f"{proc_dir}/item_features.npy"))
    train_df = pd.read_parquet(f"{proc_dir}/train.parquet")

    dataset = InteractionDataset(train_df)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    logger.info("Training samples: %d", len(dataset))

    model = TwoTower(
        user_dim=user_feats.shape[1],
        item_dim=item_feats.shape[1],
        emb_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        temperature=cfg.model.temperature,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )

    best_loss = float("inf")
    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for u_ids, i_ids in loader:
            u_feats = user_feats[u_ids].to(device)
            i_feats = item_feats[i_ids].to(device)
            loss = model.compute_loss(u_feats, i_feats)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg = total_loss / max(steps, 1)
        best_loss = min(best_loss, avg)
        logger.info(
            "epoch=%d/%d loss=%.4f lr=%.6f",
            epoch + 1,
            cfg.training.epochs,
            avg,
            scheduler.get_last_lr()[0],
        )

    out_path = Path("models_store/two_tower.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "user_dim": user_feats.shape[1],
                "item_dim": item_feats.shape[1],
                "emb_dim": cfg.model.embedding_dim,
                "hidden_dim": cfg.model.hidden_dim,
                "temperature": cfg.model.temperature,
                "dropout": cfg.model.dropout,
            },
            "best_loss": best_loss,
        },
        out_path,
    )
    logger.info("Saved Two-Tower model -> %s (best_loss=%.4f)", out_path, best_loss)
    return out_path


if __name__ == "__main__":
    train()
    record_version("models_store")
