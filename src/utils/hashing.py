"""Hashing helpers used for cache keys, audit IDs, and model versioning."""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def cache_key(prefix: str, *parts: object) -> str:
    joined = ":".join(str(p) for p in parts)
    digest = sha256_hex(joined.encode("utf-8"))[:16]
    return f"{prefix}:{digest}"
