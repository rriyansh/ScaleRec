import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Simple dot-accessible YAML config loader."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        if key not in self._data:
            raise AttributeError(f"Config key '{key}' not found")
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data


def load_config(path: str | None = None) -> Config:
    path = path or os.getenv("CONFIG_PATH", "config/config.yaml")
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(cfg_path) as f:
        return Config(yaml.safe_load(f))
