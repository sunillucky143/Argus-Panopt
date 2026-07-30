"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.inference.registry import ModelAdapterRegistry, create_default_registry
from app.api.routes.debug import router as debug_router
from app.api.routes.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.http import SecurityHeadersMiddleware


def create_app(
    settings: Settings | None = None,
    model_registry: ModelAdapterRegistry | None = None,
) -> FastAPI:
    """Build an application instance with explicit dependencies."""

    active_settings = settings or get_settings()
    application = FastAPI(
        title="Argus Panopt API",
        summary="Self-hosted document intelligence API",
        version=active_settings.app_version,
    )
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
    )
    if settings is not None:

        def settings_override() -> Settings:
            return active_settings

        application.dependency_overrides[get_settings] = settings_override
    application.state.model_registry = model_registry or create_default_registry()
    application.include_router(health_router)
    application.include_router(debug_router)
    return application


app = create_app()
