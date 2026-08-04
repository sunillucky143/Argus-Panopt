"""Typed configuration for deployment-local retrieval engines."""

from __future__ import annotations

import math
import re
from urllib.parse import urlsplit

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_EXPECTED_HOSTS = {
    "embedding_engine_endpoint": "embedding-engine",
    "reranker_engine_endpoint": "reranker-engine",
}


class Settings(BaseSettings):
    """Retrieval gateway settings loaded from ARGUS_RETRIEVAL_* variables."""

    model_config = SettingsConfigDict(
        env_prefix="ARGUS_RETRIEVAL_",
        extra="ignore",
    )

    embedding_engine_endpoint: str = "http://embedding-engine:80"
    reranker_engine_endpoint: str = "http://reranker-engine:80"
    embedding_model: str = "bge-m3"
    reranker_model: str = "bge-reranker-v2-m3"
    timeout_seconds: float = 120.0

    @field_validator("embedding_engine_endpoint", "reranker_engine_endpoint")
    @classmethod
    def validate_engine_endpoint(cls, value: str, info: ValidationInfo) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            raise ValueError("retrieval engine endpoint is invalid") from None
        expected_host = _EXPECTED_HOSTS[str(info.field_name)]
        if (
            parsed.scheme != "http"
            or parsed.hostname != expected_host
            or parsed.username is not None
            or parsed.password is not None
            or port not in (None, 80)
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("retrieval engine endpoint is invalid")
        return f"http://{expected_host}:80"

    @field_validator("embedding_model", "reranker_model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not _MODEL_PATTERN.fullmatch(value):
            raise ValueError("retrieval model name is invalid")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if not math.isfinite(value) or not 0 < value <= 300:
            raise ValueError("retrieval timeout is invalid")
        return value
