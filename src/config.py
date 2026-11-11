"""Configuration loader with env overrides, validation, and secrets handling."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    """Raised when config is invalid or missing."""


@dataclass
class AppConfig:
    name: str = "ScaleRec"
    version: str = "0.2.0"
    env: str = "development"


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    num_users: int = 500
    num_items: int = 1000
    num_interactions: int = 5000
    user_feat_dim: int = 10
    item_feat_dim: int = 10
    seed: int = 42


@dataclass
class ModelConfig:
    embedding_dim: int = 32
    hidden_dim: int = 64
    temperature: float = 0.07
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 5
    weight_decay: float = 0.0001


@dataclass
class FaissConfig:
    index_type: str = "FlatIP"
    nlist: int = 100
    m: int = 8
    nbits: int = 8
    nprobe: int = 10


@dataclass
class RankingConfig:
    n_candidates: int = 50
    top_k: int = 10
    xgb_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    request_timeout: float = 5.0
    max_batch_size: int = 100
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 120
    max_request_bytes: int = 1_048_576
    include_docs: bool = True


@dataclass
class SecurityConfig:
    api_key: str = ""
    cors_origins: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 120
    max_request_bytes: int = 1_048_576


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ttl: int = 300
    socket_timeout: float = 1.0
    max_connections: int = 50


@dataclass
class Config:
    app: AppConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    faiss: FaissConfig
    ranking: RankingConfig
    serving: ServingConfig
    security: SecurityConfig
    redis: RedisConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "app": self.app.__dict__,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "faiss": self.faiss.__dict__,
            "ranking": self.ranking.__dict__,
            "serving": self.serving.__dict__,
            "security": {
                "api_key": "***REDACTED***",
                "cors_origins": self.security.cors_origins,
                "rate_limit_per_minute": self.security.rate_limit_per_minute,
                "max_request_bytes": self.security.max_request_bytes,
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "password": "***REDACTED***" if self.redis.password else "",
                "ttl": self.redis.ttl,
                "socket_timeout": self.redis.socket_timeout,
                "max_connections": self.redis.max_connections,
            },
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if val is None:
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


def load_config(path: str | None = None) -> Config:
    """Load config from YAML with environment variable overrides."""
    cfg_path = Path(path or os.getenv("CONFIG_PATH", "config/config.yaml"))
    raw = _read_yaml(cfg_path)

    app_cfg = AppConfig(
        name=raw.get("app", {}).get("name", "ScaleRec"),
        version=raw.get("app", {}).get("version", "0.2.0"),
        env=os.getenv("APP_ENV", "development"),
    )

    data_cfg = DataConfig(**raw.get("data", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))
    faiss_cfg = FaissConfig(**raw.get("faiss", {}))
    ranking_cfg = RankingConfig(**raw.get("ranking", {}))

    serving_raw = raw.get("serving", {})
    serving_cfg = ServingConfig(
        host=os.getenv("API_HOST", serving_raw.get("host", "0.0.0.0")),
        port=int(os.getenv("API_PORT", serving_raw.get("port", 8000))),
        workers=int(os.getenv("API_WORKERS", serving_raw.get("workers", 1))),
        request_timeout=serving_raw.get("request_timeout", 5.0),
        max_batch_size=serving_raw.get("max_batch_size", 100),
        cors_origins=_env_list(
            "CORS_ORIGINS", serving_raw.get("cors_origins", ["*"])
        ),
        rate_limit_per_minute=int(
            os.getenv(
                "RATE_LIMIT_PER_MINUTE",
                serving_raw.get("rate_limit_per_minute", 120),
            )
        ),
        max_request_bytes=int(
            os.getenv(
                "MAX_REQUEST_BYTES", serving_raw.get("max_request_bytes", 1_048_576)
            )
        ),
        include_docs=_env_bool("INCLUDE_DOCS", serving_raw.get("include_docs", True)),
    )

    api_key = os.getenv("API_KEY", "")
    if not api_key and app_cfg.env == "production":
        raise ConfigError(
            "API_KEY is required in production. Set it via environment variable."
        )
    if not api_key:
        api_key = "dev-insecure-key-change-before-deploying"

    security_cfg = SecurityConfig(
        api_key=api_key,
        cors_origins=serving_cfg.cors_origins,
        rate_limit_per_minute=serving_cfg.rate_limit_per_minute,
        max_request_bytes=serving_cfg.max_request_bytes,
    )

    redis_cfg = RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD", ""),
        ttl=int(os.getenv("REDIS_TTL", "300")),
        socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "1.0")),
        max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
    )

    _validate(data_cfg, model_cfg, training_cfg, ranking_cfg)

    return Config(
        app=app_cfg,
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        faiss=faiss_cfg,
        ranking=ranking_cfg,
        serving=serving_cfg,
        security=security_cfg,
        redis=redis_cfg,
    )


def _validate(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    ranking_cfg: RankingConfig,
) -> None:
    if data_cfg.num_users <= 0:
        raise ConfigError("data.num_users must be > 0")
    if data_cfg.num_items <= 0:
        raise ConfigError("data.num_items must be > 0")
    if model_cfg.embedding_dim <= 0:
        raise ConfigError("model.embedding_dim must be > 0")
    if model_cfg.temperature <= 0:
        raise ConfigError("model.temperature must be > 0")
    if training_cfg.batch_size <= 0:
        raise ConfigError("training.batch_size must be > 0")
    if training_cfg.epochs <= 0:
        raise ConfigError("training.epochs must be > 0")
    if ranking_cfg.top_k <= 0 or ranking_cfg.n_candidates < ranking_cfg.top_k:
        raise ConfigError("ranking.top_k must be > 0 and <= n_candidates")
