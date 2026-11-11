"""Custom exception hierarchy for consistent error handling."""
from __future__ import annotations


class ScaleRecError(Exception):
    """Base exception for all ScaleRec errors."""

    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str = "", *, details: dict | None = None):
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        self.details = details or {}


class ConfigError(ScaleRecError):
    http_status = 500
    error_code = "config_error"


class ValidationError(ScaleRecError):
    http_status = 400
    error_code = "validation_error"


class AuthenticationError(ScaleRecError):
    http_status = 401
    error_code = "authentication_error"


class RateLimitError(ScaleRecError):
    http_status = 429
    error_code = "rate_limit_exceeded"


class NotFoundError(ScaleRecError):
    http_status = 404
    error_code = "not_found"


class ModelNotReadyError(ScaleRecError):
    http_status = 503
    error_code = "model_not_ready"


class CacheError(ScaleRecError):
    http_status = 500
    error_code = "cache_error"


class UpstreamError(ScaleRecError):
    http_status = 502
    error_code = "upstream_error"
