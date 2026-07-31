from fastapi.testclient import TestClient

from app.adapters.inference.fake import FakeModelAdapter
from app.adapters.inference.registry import ModelAdapterRegistry
from app.core.config import Settings
from app.main import create_app


def test_liveness_contract() -> None:
    client = TestClient(create_app(Settings(environment="test", app_version="9.8.7")))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "api", "version": "9.8.7"}


def test_readiness_contract() -> None:
    client = TestClient(
        create_app(
            Settings(
                environment="test",
                app_version="9.8.7",
                model_provider="fake",
                model_name="readiness-model",
                model_context_ceiling=4096,
            )
        )
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "api",
        "version": "9.8.7",
        "model": {
            "provider": "fake",
            "model_name": "readiness-model",
            "max_context": 4096,
            "vision": False,
            "streaming": True,
            "quantization": "none",
            "speculative_decoding": False,
            "prefix_caching": False,
        },
    }


def test_readiness_returns_capabilities_with_503_when_model_is_unavailable() -> None:
    registry = ModelAdapterRegistry()
    registry.register(
        "fake",
        lambda settings: FakeModelAdapter(
            model_name=settings.model_name,
            max_context=settings.model_context_ceiling,
            ready=False,
        ),
    )
    client = TestClient(
        create_app(
            Settings(
                environment="test",
                model_provider="fake",
                model_name="offline-model",
            ),
            model_registry=registry,
        )
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["model"]["model_name"] == "offline-model"


def test_readiness_returns_generic_503_when_adapter_is_not_registered() -> None:
    client = TestClient(
        create_app(
            Settings(environment="test", model_provider="llama_cpp"),
            model_registry=ModelAdapterRegistry(),
        )
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Local inference service is unavailable."}


def test_security_headers_and_valid_correlation_id() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.get("/health/live", headers={"X-Request-ID": "request-12345678"})

    assert response.headers["X-Request-ID"] == "request-12345678"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert (
        response.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
    )


def test_invalid_correlation_id_is_replaced() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.get("/health/live", headers={"X-Request-ID": "<script>"})

    assert response.headers["X-Request-ID"] != "<script>"
    assert len(response.headers["X-Request-ID"]) == 32


def test_openapi_is_available() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Argus Panopt API"
