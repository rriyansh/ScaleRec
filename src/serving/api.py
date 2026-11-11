"""FastAPI application entrypoint with full security stack."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.config import load_config
from src.exceptions import ScaleRecError, ValidationError
from src.logger import get_logger, set_request_id, setup_logging
from src.metrics import metrics_endpoint
from src.middleware import (
    AuditMiddleware,
    BodySizeLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    TimingMiddleware,
)
from src.security import require_api_key
from src.serving.cache import RedisCache
from src.serving.health import router as health_router
from src.serving.recommender import Recommender
from src.serving.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    ErrorResponse,
    Recommendation,
    RecommendResponse,
)

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    cfg = load_config()
    logger.info("Starting ScaleRec version=%s env=%s", cfg.app.version, cfg.app.env)

    app.state.config = cfg
    app.state.version = cfg.app.version
    app.state.start_time = time.time()

    try:
        app.state.recommender = Recommender(cfg)
    except Exception as e:
        logger.error("Failed to load recommender: %s", e)
        app.state.recommender = None

    app.state.cache = RedisCache(cfg.redis)
    logger.info("ScaleRec startup complete")
    yield
    logger.info("ScaleRec shutting down")
    app.state.recommender = None
    app.state.cache = None


def create_app() -> FastAPI:
    cfg = load_config()

    app = FastAPI(
        title=cfg.app.name,
        version=cfg.app.version,
        default_response_class=ORJSONResponse,
        docs_url="/docs" if cfg.serving.include_docs else None,
        redoc_url="/redoc" if cfg.serving.include_docs else None,
        openapi_url="/openapi.json" if cfg.serving.include_docs else None,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Middleware order: outermost first
    app.add_middleware(AuditMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=cfg.security.max_request_bytes)
    app.add_middleware(SlowAPIMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.serving.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type", "X-Request-ID"],
        max_age=600,
    )

    app.include_router(health_router)

    @app.get("/metrics", include_in_schema=False)
    async def _metrics(request: Request):
        return await metrics_endpoint(request)

    @app.exception_handler(ScaleRecError)
    async def _scale_rec_handler(request: Request, exc: ScaleRecError):
        rid = getattr(request.state, "request_id", "")
        return ORJSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(
                error=exc.error_code,
                message=exc.message,
                request_id=rid,
                details=exc.details,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "")
        logger.exception("Unhandled error path=%s", request.url.path)
        return ORJSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                message="An internal error occurred",
                request_id=rid,
            ).model_dump(),
        )

    @app.get(
        "/recommend/{user_id}",
        response_model=RecommendResponse,
        dependencies=[Depends(require_api_key)],
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    @limiter.limit(f"{cfg.security.rate_limit_per_minute}/minute")
    async def recommend(user_id: int, request: Request) -> RecommendResponse:
        recommender: Recommender | None = request.app.state.recommender
        cache: RedisCache | None = request.app.state.cache
        if recommender is None or cache is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        rid = getattr(request.state, "request_id", "")
        set_request_id(rid)

        cache_key = f"rec:{user_id}"
        cached = cache.get(cache_key)
        if cached is not None:
            try:
                recs = [Recommendation(**r) for r in cached]
                return RecommendResponse(
                    user_id=user_id, recommendations=recs, cached=True, request_id=rid
                )
            except Exception:
                cache.delete(cache_key)

        try:
            raw = recommender.recommend(user_id)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.message) from e

        cache.set(cache_key, raw)
        recs = [Recommendation(**r) for r in raw]
        return RecommendResponse(
            user_id=user_id, recommendations=recs, cached=False, request_id=rid
        )

    @app.post(
        "/recommend/batch",
        response_model=BatchRecommendResponse,
        dependencies=[Depends(require_api_key)],
    )
    @limiter.limit(f"{max(1, cfg.security.rate_limit_per_minute // 4)}/minute")
    async def recommend_batch(
        payload: BatchRecommendRequest, request: Request
    ) -> BatchRecommendResponse:
        recommender: Recommender | None = request.app.state.recommender
        cache: RedisCache | None = request.app.state.cache
        if recommender is None or cache is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        rid = getattr(request.state, "request_id", "")
        results: dict[int, list[Recommendation]] = {}
        for uid in payload.user_ids:
            key = f"rec:{uid}"
            cached = cache.get(key)
            if cached is not None:
                results[uid] = [Recommendation(**r) for r in cached]
                continue
            try:
                raw = recommender.recommend(uid)
            except ValidationError:
                results[uid] = []
                continue
            cache.set(key, raw)
            results[uid] = [Recommendation(**r) for r in raw]

        return BatchRecommendResponse(results=results, request_id=rid)

    @app.get("/", include_in_schema=False)
    async def _root() -> dict:
        return {"service": cfg.app.name, "version": cfg.app.version}

    return app


app = create_app()
