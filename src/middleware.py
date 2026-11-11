"""Custom Starlette middleware: request ID, timing, audit, security headers, size limit."""
from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from src.logger import get_logger, set_request_id
from src.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_SIZE,
)
from src.security import SECURITY_HEADERS

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID (from header or fresh) and echoes it back."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("X-Request-ID")
        request_id = incoming if _is_safe_id(incoming) else uuid.uuid4().hex
        set_request_id(request_id)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def _is_safe_id(value: str | None, max_len: int = 64) -> bool:
    if not value or len(value) > max_len:
        return False
    return all(c.isalnum() or c in {"-", "_", "."} for c in value)


class TimingMiddleware(BaseHTTPMiddleware):
    """Measures request latency and records Prometheus histogram."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        HTTP_REQUEST_DURATION.labels(
            method=request.method, path=path, status=response.status_code
        ).observe(duration)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method, path=path, status=response.status_code
        ).inc()
        response.headers["X-Process-Time"] = f"{duration:.6f}"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds standard security headers to every response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose Content-Length exceeds max bytes."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                HTTP_REQUEST_SIZE.observe(size)
                if size > self.max_bytes:
                    return JSONResponse(
                        {"detail": "Request body too large"},
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse(
                    {"detail": "Invalid Content-Length header"}, status_code=400
                )
        return await call_next(request)


class AuditMiddleware(BaseHTTPMiddleware):
    """Emits a structured audit log for every request."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        logger.info(
            "audit method=%s path=%s status=%d client=%s",
            request.method,
            request.url.path,
            response.status_code,
            request.client.host if request.client else "unknown",
        )
        return response
