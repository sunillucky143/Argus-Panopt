"""Liveness and readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_chat_model
from app.api.schemas import HealthResponse, ModelCapabilitiesResponse, ReadinessResponse
from app.core.config import Settings, get_settings
from app.ports.inference import ChatModelPort

router = APIRouter(tags=["system"])


@router.get("/health/live", response_model=HealthResponse)
async def liveness(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the API process can serve requests."""

    return HealthResponse(status="ok", version=settings.app_version)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The configured local model is unavailable.",
        }
    },
)
async def readiness(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    model: Annotated[ChatModelPort, Depends(get_chat_model)],
) -> ReadinessResponse:
    """Report API readiness and content-free active-model capabilities."""

    capabilities = await model.capabilities()
    model_ready = await model.health()
    if not model_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if model_ready else "unavailable",
        version=settings.app_version,
        model=ModelCapabilitiesResponse.from_domain(capabilities),
    )
