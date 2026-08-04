"""Deployment-internal Prometheus exposition."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from app.core.config import Settings, get_settings

router = APIRouter(tags=["internal"])


@router.get("/internal/metrics", include_in_schema=False)
def metrics(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Expose content-free process metrics only on the internal backend service."""

    if not settings.inference_metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    registry = cast(CollectorRegistry, request.app.state.prometheus_registry)
    return Response(
        content=generate_latest(registry),
        headers={"Content-Type": CONTENT_TYPE_LATEST, "Cache-Control": "no-store"},
    )
