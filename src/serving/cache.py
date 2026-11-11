"""Redis cache wrapper with circuit breaker, pooling, and metrics."""
from __future__ import annotations

import json
from typing import Any

import redis

from src.circuit_breaker import CircuitBreaker
from src.config import RedisConfig
from src.logger import get_logger
from src.metrics import CACHE_ERRORS, CACHE_HITS, CACHE_MISSES

logger = get_logger(__name__)


class RedisCache:
    def __init__(self, cfg: RedisConfig):
        self.ttl = cfg.ttl
        self._breaker = CircuitBreaker(
            name="redis", failure_threshold=5, recovery_timeout=30.0
        )
        try:
            pool = redis.ConnectionPool(
                host=cfg.host,
                port=cfg.port,
                db=cfg.db,
                password=cfg.password or None,
                socket_timeout=cfg.socket_timeout,
                socket_connect_timeout=cfg.socket_timeout,
                max_connections=cfg.max_connections,
                decode_responses=True,
            )
            self.client: redis.Redis | None = redis.Redis(connection_pool=pool)
            self.client.ping()
            self.enabled = True
            logger.info("Redis connected: %s:%d db=%d", cfg.host, cfg.port, cfg.db)
        except Exception as e:
            self.client = None
            self.enabled = False
            logger.warning("Redis unavailable, cache disabled: %s", e)

    def get(self, key: str) -> Any | None:
        if not self.enabled or self.client is None:
            return None
        try:
            val = self._breaker.call(self.client.get, key)
            if val is None:
                CACHE_MISSES.inc()
                return None
            CACHE_HITS.inc()
            return json.loads(val)
        except Exception as e:
            CACHE_ERRORS.inc()
            logger.debug("cache_get_failed key=%s err=%s", key, e)
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            payload = json.dumps(value, separators=(",", ":"))
            self._breaker.call(self.client.setex, key, self.ttl, payload)
        except Exception as e:
            CACHE_ERRORS.inc()
            logger.debug("cache_set_failed key=%s err=%s", key, e)

    def delete(self, key: str) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self._breaker.call(self.client.delete, key)
        except Exception:
            CACHE_ERRORS.inc()

    def health(self) -> bool:
        if not self.enabled or self.client is None:
            return False
        try:
            return self._breaker.call(self.client.ping) is True
        except Exception:
            return False
