import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb


class Ranker:
    """Thin wrapper around XGBoost classifier used for CTR ranking."""

    def __init__(self, params: dict):
        self.params = params
        self.model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> None:
        self.model = xgb.XGBClassifier(**self.params, use_label_encoder=False)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Ranker not trained / loaded")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save untrained ranker")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        self.model = xgb.XGBClassifier(**self.params)
        self.model.load_model(str(path))
