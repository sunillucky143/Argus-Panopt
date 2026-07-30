"""Typed environment configuration."""

from functools import lru_cache

from pydantic import Field
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

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        """Return normalized, non-empty CORS origins."""

        return tuple(origin.strip() for origin in self.cors_origins.split(",") if origin.strip())


@lru_cache
def get_settings() -> Settings:
    """Return immutable process-level settings."""

    return Settings()
