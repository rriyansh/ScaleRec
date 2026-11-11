"""Model versioning — tracks SHA256 of artifacts for reproducibility."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.hashing import sha256_file
from src.utils.io import load_json, save_json

VERSION_FILE = "version.json"


def record_version(models_dir: str | Path) -> dict:
    """Write version info for all artifacts in the models directory."""
    d = Path(models_dir)
    artifacts = {}
    for f in d.iterdir():
        if f.is_file() and f.name != VERSION_FILE:
            artifacts[f.name] = {
                "sha256": sha256_file(f),
                "size_bytes": f.stat().st_size,
            }
    version_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    save_json(version_data, d / VERSION_FILE)
    return version_data


def verify_version(models_dir: str | Path) -> bool:
    """Verify all artifacts match recorded hashes."""
    d = Path(models_dir)
    version_path = d / VERSION_FILE
    if not version_path.exists():
        return False
    recorded = load_json(version_path)
    for name, info in recorded.get("artifacts", {}).items():
        f = d / name
        if not f.exists():
            return False
        if sha256_file(f) != info["sha256"]:
            return False
    return True


def get_version(models_dir: str | Path) -> dict | None:
    path = Path(models_dir) / VERSION_FILE
    if not path.exists():
        return None
    return load_json(path)
