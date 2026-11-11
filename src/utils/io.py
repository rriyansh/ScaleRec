"""Safe IO helpers."""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write(target: Path, write_fn: Any) -> None:
    """Write to temp file then atomically move into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, prefix=".tmp_")
    os.close(fd)
    try:
        write_fn(Path(tmp_path))
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_json(obj: Any, path: str | Path) -> None:
    target = Path(path)

    def _w(p: Path) -> None:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    _atomic_write(target, _w)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_npy(arr: np.ndarray, path: str | Path) -> None:
    target = Path(path)

    def _w(p: Path) -> None:
        np.save(str(p), arr)

    _atomic_write(target, _w)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(str(path))


def save_pickle(obj: Any, path: str | Path) -> None:
    target = Path(path)

    def _w(p: Path) -> None:
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    _atomic_write(target, _w)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_text(text: str, path: str | Path) -> None:
    target = Path(path)

    def _w(p: Path) -> None:
        p.write_text(text, encoding="utf-8")

    _atomic_write(target, _w)
