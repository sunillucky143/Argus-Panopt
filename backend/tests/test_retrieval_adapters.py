import asyncio
import json
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from app.adapters.inference.retrieval import (
    _MAX_RESPONSE_BYTES,
    EmbeddingServiceAdapter,
    InvalidRetrievalRequestError,
    RerankerServiceAdapter,
    RetrievalAdapterError,
    RetrievalProtocolError,
)
from app.core.config import Settings
from app.domain.inference import EmbeddingKind, Passage


def test_embedding_adapter_translates_and_orders_vectors() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "bge-m3",
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ],
            },
        )

    adapter = EmbeddingServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-m3",
        transport=httpx.MockTransport(handler),
    )
    vectors = asyncio.run(adapter.embed(["query text", "passage text"], "passage"))

    assert captured == {
        "path": "/v1/embeddings",
        "payload": {
            "model": "bge-m3",
            "texts": ["query text", "passage text"],
            "kind": "passage",
        },
    }
    assert vectors == [(0.1, 0.2), (0.3, 0.4)]


def test_reranker_adapter_preserves_metadata_and_sorts_scores() -> None:
    captured: dict[str, Any] = {}
    passages = [
        Passage(identifier="first", text="first text", metadata={"page": "1"}),
        Passage(identifier="second", text="second text", metadata={"page": "2"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "bge-reranker-v2-m3",
                "data": [
                    {"identifier": "first", "score": 0.2},
                    {"identifier": "second", "score": 0.9},
                ],
            },
        )

    adapter = RerankerServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-reranker-v2-m3",
        transport=httpx.MockTransport(handler),
    )
    results = asyncio.run(adapter.rerank("local query", passages, top_k=2))

    assert captured == {
        "path": "/v1/rerank",
        "payload": {
            "model": "bge-reranker-v2-m3",
            "query": "local query",
            "passages": [
                {"identifier": "first", "text": "first text"},
                {"identifier": "second", "text": "second text"},
            ],
            "top_k": 2,
        },
    }
    assert [result.passage.identifier for result in results] == ["second", "first"]
    assert [result.score for result in results] == [0.9, 0.2]
    assert results[0].passage.metadata == {"page": "2"}


@pytest.mark.parametrize(
    "body",
    [
        {"model": "wrong-model", "data": [{"index": 0, "embedding": [0.1]}]},
        {"model": "bge-m3", "data": []},
        {
            "model": "bge-m3",
            "data": [
                {"index": 0, "embedding": [0.1]},
                {"index": 1, "embedding": [0.2, 0.3]},
            ],
        },
        {"model": "bge-m3", "data": [{"index": 0, "embedding": [float("nan")]}]},
        {"model": "bge-m3", "data": [{"index": True, "embedding": [0.1]}]},
    ],
)
def test_embedding_adapter_rejects_malformed_responses(body: dict[str, Any]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=json.dumps(body).encode(), headers={"content-type": "application/json"}
        )

    adapter = EmbeddingServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-m3",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RetrievalProtocolError, match="invalid response"):
        asyncio.run(adapter.embed(["first", "second"], "query"))


@pytest.mark.parametrize(
    "body",
    [
        {
            "model": "wrong-model",
            "data": [{"identifier": "first", "score": 1.0}],
        },
        {"model": "bge-reranker-v2-m3", "data": []},
        {
            "model": "bge-reranker-v2-m3",
            "data": [{"identifier": "unknown", "score": 1.0}],
        },
        {
            "model": "bge-reranker-v2-m3",
            "data": [{"identifier": "first", "score": float("inf")}],
        },
    ],
)
def test_reranker_adapter_rejects_malformed_responses(body: dict[str, Any]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=json.dumps(body).encode(), headers={"content-type": "application/json"}
        )

    adapter = RerankerServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-reranker-v2-m3",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RetrievalProtocolError, match="invalid response"):
        asyncio.run(
            adapter.rerank(
                "query",
                [Passage(identifier="first", text="text")],
                top_k=1,
            )
        )


def test_adapter_wraps_http_failures_without_echoing_content() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="private document content")

    adapter = EmbeddingServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-m3",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RetrievalAdapterError) as caught:
        asyncio.run(adapter.embed(["private input"], "query"))

    assert str(caught.value) == "Local retrieval service request failed."
    assert "private" not in str(caught.value)


def test_adapter_does_not_follow_redirects() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(307, headers={"location": "https://example.com/collect"})

    adapter = EmbeddingServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-m3",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RetrievalAdapterError, match="request failed"):
        asyncio.run(adapter.embed(["private input"], "query"))

    assert paths == ["/v1/embeddings"]


def test_adapter_rejects_oversized_response_before_parsing() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * (_MAX_RESPONSE_BYTES + 1),
            headers={"content-type": "application/json"},
        )

    adapter = EmbeddingServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-m3",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RetrievalProtocolError, match="invalid response"):
        asyncio.run(adapter.embed(["private input"], "query"))


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com",
        "http://8.8.8.8:8081",
        "http://user:secret@embedding-service:8081",
        "http://embedding-service:notaport",
        "http://-embedding-service:8081",
        "http://embedding-service-:8081",
    ],
)
def test_adapters_reject_non_local_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError, match="retrieval endpoint"):
        EmbeddingServiceAdapter(endpoint=endpoint, model_name="bge-m3")


@pytest.mark.parametrize("model_name", ["", "   "])
def test_adapters_reject_empty_model_names(model_name: str) -> None:
    with pytest.raises(ValueError, match="model name"):
        EmbeddingServiceAdapter(
            endpoint="http://embedding-service:8081",
            model_name=model_name,
        )


def test_embedding_adapter_rejects_invalid_requests_without_network() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be called")

    adapter = EmbeddingServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-m3",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(InvalidRetrievalRequestError):
        asyncio.run(adapter.embed([], "query"))
    with pytest.raises(InvalidRetrievalRequestError):
        asyncio.run(adapter.embed(["text"], cast(EmbeddingKind, "document")))


def test_reranker_adapter_rejects_invalid_requests_without_network() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be called")

    adapter = RerankerServiceAdapter(
        endpoint="http://embedding-service:8081",
        model_name="bge-reranker-v2-m3",
        transport=httpx.MockTransport(handler),
    )
    duplicate_passages = [
        Passage(identifier="same", text="first"),
        Passage(identifier="same", text="second"),
    ]

    with pytest.raises(InvalidRetrievalRequestError):
        asyncio.run(adapter.rerank("query", duplicate_passages, top_k=1))
    with pytest.raises(InvalidRetrievalRequestError):
        asyncio.run(
            adapter.rerank(
                "query",
                [Passage(identifier="one", text="text")],
                top_k=2,
            )
        )
    with pytest.raises(InvalidRetrievalRequestError):
        asyncio.run(
            adapter.rerank(
                "query",
                [Passage(identifier=cast(str, 7), text="text")],
                top_k=1,
            )
        )
    with pytest.raises(InvalidRetrievalRequestError):
        asyncio.run(
            adapter.rerank(
                "query",
                [Passage(identifier="one", text="text")],
                top_k=cast(int, 1.5),
            )
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://embedding-service:8081",
        "http://localhost:8081",
        "http://10.20.30.40:8081",
    ],
)
def test_embedding_endpoint_configuration_accepts_local_hosts(endpoint: str) -> None:
    assert Settings(embedding_endpoint=endpoint).embedding_endpoint == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com",
        "http://8.8.8.8:8081",
        "ftp://embedding-service:8081",
        "http://embedding-service:notaport",
    ],
)
def test_embedding_endpoint_configuration_rejects_public_hosts(endpoint: str) -> None:
    with pytest.raises(ValidationError, match="embedding endpoint"):
        Settings(embedding_endpoint=endpoint)
