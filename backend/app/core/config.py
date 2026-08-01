"""Typed environment configuration."""

from functools import lru_cache

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.urls import validate_internal_service_endpoint


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
    embedding_endpoint: str = "http://embedding-service:8081"
    embedding_model: str = "bge-m3"
    reranker_model: str = "bge-reranker-v2-m3"

    @field_validator("model_endpoint", "embedding_endpoint")
    @classmethod
    def validate_local_service_endpoint(cls, value: str, info: ValidationInfo) -> str:
        """Reject endpoints that could route PHI outside the deployment."""

        label = "model endpoint" if info.field_name == "model_endpoint" else "embedding endpoint"
        return validate_internal_service_endpoint(value, label=label)

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        """Return normalized, non-empty CORS origins."""

        return tuple(origin.strip() for origin in self.cors_origins.split(",") if origin.strip())


@lru_cache
def get_settings() -> Settings:
    """Return immutable process-level settings."""

    return Settings()
