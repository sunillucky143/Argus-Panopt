"""API response contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.domain.inference import ModelCapabilities


class ServiceResponse(BaseModel):
    """Fields shared by process liveness and dependency readiness."""

    model_config = ConfigDict(extra="forbid")

    service: Literal["api"] = "api"
    version: str


class HealthResponse(ServiceResponse):
    """Service health information with a stable, strict schema."""

    status: Literal["ok", "ready"]


class ModelCapabilitiesResponse(BaseModel):
    """Public, content-free details about the active local model."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model_name: str
    max_context: int
    vision: bool
    streaming: bool
    quantization: str
    speculative_decoding: bool
    prefix_caching: bool

    @classmethod
    def from_domain(cls, capabilities: ModelCapabilities) -> "ModelCapabilitiesResponse":
        """Map provider-neutral domain capabilities to the API contract."""

        return cls(
            provider=capabilities.provider,
            model_name=capabilities.model_name,
            max_context=capabilities.max_context,
            vision=capabilities.vision,
            streaming=capabilities.streaming,
            quantization=capabilities.quantization,
            speculative_decoding=capabilities.speculative_decoding,
            prefix_caching=capabilities.prefix_caching,
        )


class ReadinessResponse(ServiceResponse):
    """API and active-model readiness for Compose and administration."""

    status: Literal["ready", "unavailable"]
    model: ModelCapabilitiesResponse
