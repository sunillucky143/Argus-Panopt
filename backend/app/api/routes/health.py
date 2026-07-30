"""Liveness and readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.schemas import HealthResponse
from app.core.config import Settings, get_settings

router = APIRouter(tags=["system"])


@router.get("/health/live", response_model=HealthResponse)
async def liveness(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the API process can serve requests."""

    return HealthResponse(status="ok", version=settings.app_version)


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report readiness; dependency checks will be added with their phases."""

    return HealthResponse(status="ready", version=settings.app_version)
