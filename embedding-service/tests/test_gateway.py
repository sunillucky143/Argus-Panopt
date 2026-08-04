from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app


def _client(
    embedding_handler: httpx.MockTransport,
    reranker_handler: httpx.MockTransport,
) -> TestClient:
    return TestClient(
        create_app(
            Settings(),
            embedding_transport=embedding_handler,
            reranker_transport=reranker_handler,
        )
    )


def _healthy(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "ready"})


def test_embedding_contract_translates_to_tei_without_content_leakage() -> None:
    captured: dict[str, Any] = {}

    def embedding_handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=[[0.1, 0.2], [0.3, 0.4]])

    with _client(httpx.MockTransport(embedding_handler), httpx.MockTransport(_healthy)) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-m3", "texts": ["private one", "private two"], "kind": "query"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "bge-m3",
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ],
    }
    assert captured == {
        "path": "/embed",
        "body": {
            "inputs": ["private one", "private two"],
            "normalize": True,
            "truncate": True,
        },
    }


def test_rerank_contract_maps_indexes_to_opaque_identifiers() -> None:
    captured: dict[str, Any] = {}

    def reranker_handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=[
                {"index": 1, "score": 0.2},
                {"index": 2, "score": 0.9},
                {"index": 0, "score": 0.5},
            ],
        )

    with _client(httpx.MockTransport(_healthy), httpx.MockTransport(reranker_handler)) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "model": "bge-reranker-v2-m3",
                "query": "private query",
                "passages": [
                    {"identifier": "p1", "text": "first"},
                    {"identifier": "p2", "text": "second"},
                    {"identifier": "p3", "text": "third"},
                ],
                "top_k": 2,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "bge-reranker-v2-m3",
        "data": [
            {"identifier": "p3", "score": 0.9},
            {"identifier": "p1", "score": 0.5},
        ],
    }
    assert captured == {
        "path": "/rerank",
        "body": {
            "query": "private query",
            "texts": ["first", "second", "third"],
            "truncate": True,
            "raw_scores": False,
            "return_text": False,
        },
    }


def test_readiness_requires_both_local_workers() -> None:
    def unavailable(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with _client(httpx.MockTransport(_healthy), httpx.MockTransport(unavailable)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "embedding_ready": True,
        "reranker_ready": False,
    }


def test_live_does_not_depend_on_model_workers() -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private detail")

    with _client(httpx.MockTransport(fail), httpx.MockTransport(fail)) as client:
        assert client.get("/health/live").json() == {"status": "live"}


@pytest.mark.parametrize(
    ("endpoint", "field"),
    [
        ("https://embedding-engine", "embedding_engine_endpoint"),
        ("http://example.com", "embedding_engine_endpoint"),
        ("http://user@embedding-engine", "embedding_engine_endpoint"),
        ("http://reranker-engine/path", "reranker_engine_endpoint"),
    ],
)
def test_settings_reject_noncanonical_engine_endpoints(endpoint: str, field: str) -> None:
    with pytest.raises(ValidationError, match="retrieval engine endpoint"):
        Settings.model_validate({field: endpoint})


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "wrong", "texts": ["secret"], "kind": "query"},
        {"model": "bge-m3", "texts": ["   "], "kind": "query"},
        {"model": "bge-m3", "texts": ["secret"], "kind": "wrong"},
    ],
)
def test_invalid_embedding_requests_are_content_safe(payload: dict[str, object]) -> None:
    with _client(httpx.MockTransport(_healthy), httpx.MockTransport(_healthy)) as client:
        response = client.post("/v1/embeddings", json=payload)

    assert response.status_code in {400, 422}
    assert "secret" not in response.text
    assert "wrong" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        b"[[0.1], [0.2, 0.3]]",
        b"[[NaN]]",
        b'{"not": "a list"}',
    ],
)
def test_embedding_rejects_malformed_engine_responses(body: bytes) -> None:
    def malformed(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    with _client(httpx.MockTransport(malformed), httpx.MockTransport(_healthy)) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-m3", "texts": ["one"], "kind": "passage"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Local retrieval engine is unavailable."}


def test_rerank_rejects_duplicates_and_invalid_top_k_without_network() -> None:
    def forbidden(_: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be called")

    with _client(httpx.MockTransport(forbidden), httpx.MockTransport(forbidden)) as client:
        duplicate = client.post(
            "/v1/rerank",
            json={
                "model": "bge-reranker-v2-m3",
                "query": "query",
                "passages": [
                    {"identifier": "same", "text": "one"},
                    {"identifier": "same", "text": "two"},
                ],
                "top_k": 1,
            },
        )
        excessive = client.post(
            "/v1/rerank",
            json={
                "model": "bge-reranker-v2-m3",
                "query": "query",
                "passages": [{"identifier": "one", "text": "one"}],
                "top_k": 2,
            },
        )

    assert duplicate.status_code == 400
    assert excessive.status_code == 400


def test_engine_transport_errors_are_generic() -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sensitive local address")

    with _client(httpx.MockTransport(fail), httpx.MockTransport(_healthy)) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-m3", "texts": ["secret text"], "kind": "query"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Local retrieval engine is unavailable."}
    assert "secret" not in response.text
    assert "sensitive" not in response.text
