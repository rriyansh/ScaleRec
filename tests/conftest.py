import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["CONFIG_PATH"] = str(ROOT / "config" / "config.test.yaml")


@pytest.fixture(scope="session")
def cfg_path() -> str:
    return str(ROOT / "config" / "config.test.yaml")
