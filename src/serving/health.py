"""Liveness, readiness, and startup probes."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

from src.logger import get_logger
from src.serving.schemas import HealthResponse, ReadinessResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

_START_TIME = time.time()


@router.get("/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Kubernetes liveness probe: process is up and responsive."""
    return HealthResponse(
        status="ok", uptime_seconds=time.time() - _START_TIME
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Kubernetes readiness probe: all dependencies are usable."""
    app_state = getattr(request.app.state, "recommender", None)
    cache = getattr(request.app.state, "cache", None)
    version = getattr(request.app.state, "version", "")

    checks: dict[str, bool] = {
        "recommender": app_state is not None,
        "cache": cache is not None and cache.health(),
    }
    if app_state is not None:
        checks.update(app_state.health_check())

    ok = all(v for k, v in checks.items() if k != "cache")
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ok" if ok else "degraded", checks=checks, version=version
    )


@router.get("/startup", response_model=HealthResponse)
async def startup(request: Request) -> HealthResponse:
    """Startup probe: model fully loaded and ready."""
    loaded = getattr(request.app.state, "recommender", None) is not None
    return HealthResponse(
        status="ok" if loaded else "loading",
        uptime_seconds=time.time() - _START_TIME,
    )
