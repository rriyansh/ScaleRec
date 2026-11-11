"""Pydantic schemas with strict validation for all API I/O."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Recommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: int = Field(..., ge=0)
    score: float = Field(..., ge=0.0, le=1.0)


class RecommendResponse(BaseModel):
    user_id: int = Field(..., ge=0)
    recommendations: list[Recommendation]
    cached: bool
    request_id: str = ""


class BatchRecommendRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_ids: list[int] = Field(..., min_length=1, max_length=100)

    def validate_user_ids(self, max_users: int) -> None:
        for uid in self.user_ids:
            if uid < 0 or uid >= max_users:
                raise ValueError(
                    f"user_id {uid} out of range [0, {max_users})"
                )


class BatchRecommendResponse(BaseModel):
    results: dict[int, list[Recommendation]]
    request_id: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str = ""
    uptime_seconds: float = 0.0


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]
    version: str = ""


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
