import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CONFIG_PATH", str(ROOT / "config" / "config.test.yaml"))
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "1")  # unreachable → cache disabled


@pytest.fixture(scope="session")
def cfg_path() -> str:
    return str(ROOT / "config" / "config.test.yaml")


@pytest.fixture(scope="session", autouse=True)
def prepare_artifacts(cfg_path, tmp_path_factory):
    """Build all artifacts once for the session."""
    from src.data.synthetic import generate
    from src.data.preprocess import preprocess

    generate(cfg_path)
    preprocess(cfg_path)

    try:
        import faiss  # noqa: F401
    except ImportError:
        pytest.skip("faiss not installed", allow_module_level=True)

    from src.retrieval.train import train as train_retrieval
    from src.retrieval.build_index import build_index
    from src.ranking.train import train as train_ranking

    train_retrieval(cfg_path)
    build_index(cfg_path)
    train_ranking(cfg_path)
    return cfg_path
