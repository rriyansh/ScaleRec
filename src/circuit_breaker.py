"""Simple, thread-safe circuit breaker for external dependencies."""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any, Callable, TypeVar

from src.logger import get_logger
from src.metrics import CIRCUIT_STATE

logger = get_logger(__name__)

T = TypeVar("T")


class State(Enum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreaker:
    """
    Circuit breaker pattern:
      - CLOSED: normal operation
      - OPEN: fail fast (cooldown)
      - HALF_OPEN: allow trial calls to probe recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = State.CLOSED
        self._failures = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._update_metric()

    def _update_metric(self) -> None:
        try:
            CIRCUIT_STATE.labels(name=self.name).set(self._state.value)
        except Exception:
            pass

    def _should_attempt_reset(self) -> bool:
        return (
            self._state == State.OPEN
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        )

    def _transition_to(self, state: State) -> None:
        if state != self._state:
            logger.info(
                "circuit_breaker_transition name=%s from=%s to=%s",
                self.name, self._state.name, state.name,
            )
            self._state = state
            if state == State.HALF_OPEN:
                self._half_open_calls = 0
            self._update_metric()

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        with self._lock:
            if self._state == State.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(State.HALF_OPEN)
                else:
                    raise RuntimeError(f"Circuit '{self.name}' is OPEN")

            if self._state == State.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise RuntimeError(f"Circuit '{self.name}' half-open limit")
                self._half_open_calls += 1

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._transition_to(State.CLOSED)

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= self.failure_threshold:
                self._transition_to(State.OPEN)

    @property
    def state(self) -> State:
        return self._state
