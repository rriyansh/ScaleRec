"""TorchServe custom handler for the full recommendation pipeline."""
from __future__ import annotations

import json

from ts.torch_handler.base_handler import BaseHandler

from src.config import load_config
from src.logger import get_logger, setup_logging
from src.serving.cache import RedisCache
from src.serving.recommender import Recommender

logger = get_logger(__name__)


class ScaleRecHandler(BaseHandler):
    def initialize(self, ctx):
        setup_logging()
        cfg = load_config()
        self.recommender = Recommender(cfg)
        self.cache = RedisCache(cfg.redis)
        self.initialized = True
        logger.info("TorchServe handler ready")

    def preprocess(self, data):
        row = data[0]
        payload = row.get("body") or row.get("data") or row
        if isinstance(payload, (bytes, str)):
            payload = json.loads(payload)
        user_id = int(payload["user_id"])
        return user_id

    def inference(self, user_id: int):
        key = f"rec:{user_id}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        recs = self.recommender.recommend(user_id)
        self.cache.set(key, recs)
        return recs

    def postprocess(self, output):
        return [json.dumps({"recommendations": output})]
