"""Feature engineering for the XGBoost ranker."""
import numpy as np


def build_features(user_emb: np.ndarray,
                   candidate_item_embs: np.ndarray) -> np.ndarray:
    """
    Build ranker features for a (user, candidate items) pair.

    Args:
        user_emb: [D] L2-normalized user embedding
        candidate_item_embs: [K, D] L2-normalized candidate embeddings

    Returns:
        [K, 2D+1] feature matrix: [user_emb | item_emb | dot_product]
    """
    K = candidate_item_embs.shape[0]
    user_rep = np.repeat(user_emb[None, :], K, axis=0)
    dot = (user_rep * candidate_item_embs).sum(axis=1, keepdims=True)
    return np.hstack([user_rep, candidate_item_embs, dot]).astype("float32")
