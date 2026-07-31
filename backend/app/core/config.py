"""Typed environment configuration."""

import re
from functools import lru_cache
from ipaddress import ip_address
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process configuration loaded exclusively from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ARGUS_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="development", pattern=r"^(development|test|production)$")
    app_version: str = "0.1.0"
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    cors_origins: str = "http://localhost:8080"
    debug_inference_enabled: bool = False
    model_provider: str = Field(default="llama_cpp", pattern=r"^(fake|llama_cpp|vllm)$")
    model_name: str = "local-model-not-downloaded"
    model_endpoint: str = "http://inference-cpu:8080/v1"
    model_context_ceiling: int = Field(default=32_768, ge=1024, le=131_072)

    @field_validator("model_endpoint")
    @classmethod
    def validate_local_model_endpoint(cls, value: str) -> str:
        """Reject model endpoints that could route prompts outside the deployment."""

        endpoint = urlsplit(value)
        hostname = endpoint.hostname
        if (
            endpoint.scheme not in {"http", "https"}
            or hostname is None
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or any(segment in {".", ".."} for segment in endpoint.path.split("/"))
        ):
            raise ValueError("model endpoint must be a simple internal HTTP(S) URL")

        try:
            address = ip_address(hostname)
        except ValueError:
            is_internal_name = hostname == "localhost" or (
                re.fullmatch(
                    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
                    hostname,
                    flags=re.IGNORECASE,
                )
                and not hostname.startswith("-")
                and not hostname.endswith("-")
            )
            if not is_internal_name:
                raise ValueError("model endpoint hostname must be local or private") from None
        else:
            if not (address.is_loopback or address.is_private or address.is_link_local):
                raise ValueError("model endpoint hostname must be local or private")

        return value.rstrip("/")

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        """Return normalized, non-empty CORS origins."""

        return tuple(origin.strip() for origin in self.cors_origins.split(",") if origin.strip())


@lru_cache
def get_settings() -> Settings:
    """Return immutable process-level settings."""

    return Settings()
