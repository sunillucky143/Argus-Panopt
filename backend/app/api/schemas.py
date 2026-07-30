"""API response contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Service health information with a stable, strict schema."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "ready"]
    service: Literal["api"] = "api"
    version: str
