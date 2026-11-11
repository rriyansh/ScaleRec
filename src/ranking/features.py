"""Feature engineering for the XGBoost ranker."""
from __future__ import annotations

import numpy as np

from src.exceptions import ValidationError


def build_features(
    user_emb: np.ndarray, candidate_item_embs: np.ndarray
) -> np.ndarray:
    """Construct ranker features: [user_emb | item_emb | dot_product]."""
    if user_emb.ndim != 1:
        raise ValidationError("user_emb must be 1D")
    if candidate_item_embs.ndim != 2:
        raise ValidationError("candidate_item_embs must be 2D")
    if candidate_item_embs.shape[1] != user_emb.shape[0]:
        raise ValidationError("Embedding dims mismatch")

    K = candidate_item_embs.shape[0]
    user_rep = np.repeat(user_emb[None, :], K, axis=0)
    dot = (user_rep * candidate_item_embs).sum(axis=1, keepdims=True)
    return np.hstack([user_rep, candidate_item_embs, dot]).astype("float32")
