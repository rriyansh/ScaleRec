"""Data validation utilities — fail fast on schema violations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.exceptions import ValidationError


def validate_features(
    arr: np.ndarray, expected_rows: int, expected_cols: int, name: str
) -> None:
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"{name} features must be ndarray")
    if arr.ndim != 2:
        raise ValidationError(f"{name} features must be 2D, got {arr.ndim}D")
    if arr.shape != (expected_rows, expected_cols):
        raise ValidationError(
            f"{name} features shape mismatch: expected "
            f"({expected_rows}, {expected_cols}), got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} features contain NaN or Inf")


def validate_interactions(
    df: pd.DataFrame, num_users: int, num_items: int, name: str
) -> None:
    required = {"user_id", "item_id", "clicked"}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(f"{name} missing columns: {missing}")
    if df.empty:
        raise ValidationError(f"{name} is empty")
    if df["user_id"].min() < 0 or df["user_id"].max() >= num_users:
        raise ValidationError(f"{name} user_id out of range [0, {num_users})")
    if df["item_id"].min() < 0 or df["item_id"].max() >= num_items:
        raise ValidationError(f"{name} item_id out of range [0, {num_items})")
    if not df["clicked"].isin({0, 1}).all():
        raise ValidationError(f"{name} clicked must be 0 or 1")
