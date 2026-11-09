import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(config_path: str = "config/logging.yaml") -> None:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
