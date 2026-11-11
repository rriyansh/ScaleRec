from src.circuit_breaker import CircuitBreaker, State
from src.config import RedisConfig
from src.serving.cache import RedisCache


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)

    def fail():
        raise RuntimeError("boom")

    for _ in range(2):
        try:
            cb.call(fail)
        except RuntimeError:
            pass
    assert cb.state == State.OPEN


def test_cache_disabled_when_redis_unreachable():
    cfg = RedisConfig(host="invalid", port=1, socket_timeout=0.1)
    cache = RedisCache(cfg)
    assert cache.enabled is False
    assert cache.get("x") is None
    cache.set("x", "y")
    assert cache.health() is False
