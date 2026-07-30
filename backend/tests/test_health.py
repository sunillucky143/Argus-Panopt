from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_liveness_contract() -> None:
    client = TestClient(create_app(Settings(environment="test", app_version="9.8.7")))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "api", "version": "9.8.7"}


def test_readiness_contract() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


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
