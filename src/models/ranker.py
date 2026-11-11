"""XGBoost ranker wrapper with strict validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from src.exceptions import ValidationError


class Ranker:
    def __init__(self, params: dict[str, Any]):
        self.params = dict(params)
        self.model: xgb.XGBClassifier | None = None

    def _require_model(self) -> xgb.XGBClassifier:
        if self.model is None:
            raise RuntimeError("Ranker not trained or loaded")
        return self.model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        if X.size == 0 or y.size == 0:
            raise ValidationError("Cannot train on empty data")
        if X.shape[0] != y.shape[0]:
            raise ValidationError("X and y row count mismatch")
        if not np.isfinite(X).all():
            raise ValidationError("X contains NaN or Inf")
        if not np.isin(y, [0, 1]).all():
            raise ValidationError("y must contain only 0/1")

        params = {k: v for k, v in self.params.items() if k != "use_label_encoder"}
        self.model = xgb.XGBClassifier(**params, use_label_encoder=False)
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        m = self._require_model()
        if X.ndim != 2:
            raise ValidationError("X must be 2D")
        return m.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        m = self._require_model()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        m.save_model(str(path))

    def load(self, path: str | Path) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Ranker model not found: {path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        m = self._require_model()
        score = m.get_booster().get_score(importance_type=importance_type)
        return dict(score)
