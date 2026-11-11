"""Security primitives: API key auth, security headers, rate limit helpers."""
from __future__ import annotations

import hmac
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from src.config import Config, load_config
from src.logger import get_logger

logger = get_logger(__name__)


def _get_config() -> Config:
    return load_config()


class APIKeyAuth:
    """Bearer-token / API-key authentication with constant-time comparison."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("api_key cannot be empty")
        self._api_key = api_key.encode("utf-8")

    def verify(self, provided: str | None) -> bool:
        if not provided:
            return False
        try:
            return hmac.compare_digest(
                self._api_key, provided.encode("utf-8")
            )
        except Exception:
            return False

    def extract_from_header(self, authorization: str | None) -> str | None:
        """Accepts both `Bearer <key>` and raw `<key>`."""
        if not authorization:
            return None
        parts = authorization.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return authorization.strip()


def get_api_key_auth(cfg: Config | None = None) -> APIKeyAuth:
    cfg = cfg or _get_config()
    return APIKeyAuth(cfg.security.api_key)


async def require_api_key(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """FastAPI dependency that enforces API key authentication."""
    auth = get_api_key_auth()
    token = auth.extract_from_header(authorization) or x_api_key
    if not auth.verify(token):
        logger.warning(
            "auth_failed path=%s client=%s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


def generate_api_key(length: int = 48) -> str:
    """Utility to generate secure API keys (for ops / CLI)."""
    return secrets.token_urlsafe(length)


__all__ = [
    "APIKeyAuth",
    "require_api_key",
    "get_api_key_auth",
    "SECURITY_HEADERS",
    "generate_api_key",
    "Depends",
]
