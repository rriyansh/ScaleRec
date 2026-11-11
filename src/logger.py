"""Structured logging with request correlation IDs."""
from __future__ import annotations

import logging
import logging.config
import sys
from contextvars import ContextVar
from pathlib import Path

import yaml

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    _request_id_ctx.set(request_id)


def get_request_id() -> str:
    return _request_id_ctx.get()


class RequestIDFilter(logging.Filter):
    """Injects current request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


class SafeFormatter(logging.Formatter):
    """Formatter that survives missing attributes."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_ctx.get()
        return super().format(record)


_CONFIGURED = False


def setup_logging(config_path: str | None = None) -> None:
    """Configure logging once from YAML, falling back to basic config."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    path = Path(config_path or os_get("LOG_CONFIG", "config/logging.yaml"))
    if path.exists():
        with open(path) as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            SafeFormatter(
                "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s"
            )
        )
        handler.addFilter(RequestIDFilter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(os_get("LOG_LEVEL", "INFO"))

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        if not any(isinstance(f, RequestIDFilter) for f in logger.filters):
            logger.addFilter(RequestIDFilter())

    _CONFIGURED = True


def os_get(name: str, default: str) -> str:
    import os
    return os.getenv(name, default)


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
