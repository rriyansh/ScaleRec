"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import load_config
from src.logger import get_logger, setup_logging
from src.serving.cache import RedisCache
from src.serving.recommender import Recommender

logger = get_logger(__name__)
recommender: Recommender | None = None
cache: RedisCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender, cache
    setup_logging()
    cfg = load_config()
    logger.info("Loading recommender...")
    recommender = Recommender(cfg)
    cache = RedisCache(
        host=cfg.serving.redis_host,
        port=cfg.serving.redis_port,
        ttl=cfg.serving.cache_ttl,
    )
    logger.info("API ready")
    yield


app = FastAPI(title="ScaleRec", version="0.1.0", lifespan=lifespan)


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[dict]
    cached: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int) -> RecommendResponse:
    if recommender is None or cache is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    cache_key = f"rec:{user_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return RecommendResponse(user_id=user_id, recommendations=cached,
                                 cached=True)

    try:
        recs = recommender.recommend(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cache.set(cache_key, recs)
    return RecommendResponse(user_id=user_id, recommendations=recs, cached=False)
