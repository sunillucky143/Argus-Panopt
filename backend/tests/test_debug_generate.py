from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _client(
    *,
    enabled: bool = True,
    ceiling: int = 4096,
    provider: str = "fake",
    environment: str = "test",
) -> TestClient:
    return TestClient(
        create_app(
            Settings(
                environment=environment,
                debug_inference_enabled=enabled,
                model_provider=provider,
                model_name="test-fake",
                model_context_ceiling=ceiling,
            )
        )
    )


def test_debug_generate_streams_normalized_sse_without_caching() -> None:
    with _client().stream(
        "POST",
        "/v1/debug/generate",
        json={"messages": [{"role": "user", "content": "Do not log this prompt."}]},
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert lines[0] == "event: token"
    assert '"text":"Local "' in lines[1]
    assert lines[-2] == "event: complete"
    assert '"finish_reason":"stop"' in lines[-1]


def test_debug_generate_is_hidden_when_disabled() -> None:
    response = _client(enabled=False, provider="llama_cpp").post(
        "/v1/debug/generate",
        json={"messages": [{"role": "user", "content": "status"}]},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Not found."}


def test_debug_generate_enforces_configured_context_ceiling() -> None:
    response = _client(ceiling=1024).post(
        "/v1/debug/generate",
        json={
            "messages": [{"role": "user", "content": "status"}],
            "max_tokens": 1025,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Requested output exceeds the configured context ceiling."


def test_debug_generate_rejects_empty_messages() -> None:
    response = _client().post("/v1/debug/generate", json={"messages": []})

    assert response.status_code == 422


def test_debug_generate_returns_generic_error_for_unavailable_local_adapter() -> None:
    response = _client(provider="llama_cpp").post(
        "/v1/debug/generate",
        json={"messages": [{"role": "user", "content": "status"}]},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Local inference service is unavailable."}


def test_debug_generate_is_hidden_in_production_even_when_enabled() -> None:
    response = _client(enabled=True, environment="production").post(
        "/v1/debug/generate",
        json={"messages": [{"role": "user", "content": "status"}]},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Not found."}
