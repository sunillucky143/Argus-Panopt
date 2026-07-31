"""FastAPI dependency adapters."""

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status

from app.adapters.inference.registry import ModelAdapterRegistry, ModelAdapterUnavailableError
from app.core.config import Settings, get_settings
from app.ports.inference import ChatModelPort


def get_chat_model(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatModelPort:
    """Resolve the configured local model for operational and API dependencies."""

    registry = cast(ModelAdapterRegistry, request.app.state.model_registry)
    try:
        return registry.resolve(settings)
    except ModelAdapterUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local inference service is unavailable.",
        ) from error


def get_debug_chat_model(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatModelPort:
    """Resolve the diagnostic adapter only when explicitly enabled."""

    if settings.environment == "production" or not settings.debug_inference_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")

    return get_chat_model(request, settings)
