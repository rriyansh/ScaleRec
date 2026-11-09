"""Redis cache wrapper with graceful fallback."""
import json
import logging
from typing import Any

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, host: str = "localhost", port: int = 6379,
                 ttl: int = 300):
        self.ttl = ttl
        try:
            self.client = redis.Redis(
                host=host, port=port, decode_responses=True,
                socket_connect_timeout=1,
            )
            self.client.ping()
            self.enabled = True
            logger.info("Redis cache enabled at %s:%s", host, port)
        except Exception as e:
            self.client = None
            self.enabled = False
            logger.warning("Redis unavailable (%s); cache disabled", e)

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            val = self.client.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            self.client.setex(key, self.ttl, json.dumps(value))
        except Exception:
            pass
