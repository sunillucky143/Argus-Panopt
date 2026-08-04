"""Translate the Argus retrieval contract to local TEI workers."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import httpx

from app.config import Settings
from app.schemas import EmbeddingRequest, RerankRequest

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_VECTOR_DIMENSIONS = 4096


class RetrievalEngineError(RuntimeError):
    """A local retrieval engine request failed without exposing content."""


class RetrievalEngineProtocolError(RetrievalEngineError):
    """A local retrieval engine returned a malformed response."""


class RetrievalRequestError(ValueError):
    """A request violates a cross-field gateway constraint."""


def _protocol_error() -> RetrievalEngineProtocolError:
    return RetrievalEngineProtocolError("Local retrieval engine returned invalid data.")


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _protocol_error()
    number = float(value)
    if not math.isfinite(number):
        raise _protocol_error()
    return number


class RetrievalGateway:
    """Bounded clients for the local embedding and reranking workers."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_transport: httpx.AsyncBaseTransport | None = None,
        reranker_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        timeout = httpx.Timeout(settings.timeout_seconds)
        common: dict[str, Any] = {
            "timeout": timeout,
            "follow_redirects": False,
            "trust_env": False,
        }
        self.settings = settings
        self.embedding_client = httpx.AsyncClient(
            base_url=settings.embedding_engine_endpoint,
            transport=embedding_transport,
            **common,
        )
        self.reranker_client = httpx.AsyncClient(
            base_url=settings.reranker_engine_endpoint,
            transport=reranker_transport,
            **common,
        )

    async def close(self) -> None:
        await asyncio.gather(
            self.embedding_client.aclose(),
            self.reranker_client.aclose(),
        )

    async def readiness(self) -> tuple[bool, bool]:
        return await asyncio.gather(
            self._healthy(self.embedding_client),
            self._healthy(self.reranker_client),
        )

    @staticmethod
    async def _healthy(client: httpx.AsyncClient) -> bool:
        try:
            async with client.stream("GET", "/health") as response:
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    async def _post_json(
        client: httpx.AsyncClient,
        path: str,
        payload: dict[str, Any],
    ) -> object:
        chunks: list[bytes] = []
        try:
            async with client.stream("POST", path, json=payload) as response:
                response.raise_for_status()
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > _MAX_RESPONSE_BYTES:
                        raise _protocol_error()
                    chunks.append(chunk)
        except RetrievalEngineProtocolError:
            raise
        except httpx.HTTPError as error:
            raise RetrievalEngineError("Local retrieval engine request failed.") from error
        try:
            return json.loads(b"".join(chunks))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _protocol_error() from error

    async def embed(self, request: EmbeddingRequest) -> dict[str, object]:
        if request.model != self.settings.embedding_model:
            raise RetrievalRequestError("embedding model is unavailable")
        body = await self._post_json(
            self.embedding_client,
            "/embed",
            {
                "inputs": request.texts,
                "normalize": True,
                "truncate": True,
            },
        )
        if not isinstance(body, list) or len(body) != len(request.texts):
            raise _protocol_error()

        dimension: int | None = None
        data: list[dict[str, object]] = []
        for index, raw_vector in enumerate(body):
            if (
                not isinstance(raw_vector, list)
                or not raw_vector
                or len(raw_vector) > _MAX_VECTOR_DIMENSIONS
            ):
                raise _protocol_error()
            vector = [_finite_number(value) for value in raw_vector]
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise _protocol_error()
            data.append({"index": index, "embedding": vector})
        return {"model": self.settings.embedding_model, "data": data}

    async def rerank(self, request: RerankRequest) -> dict[str, object]:
        if request.model != self.settings.reranker_model:
            raise RetrievalRequestError("reranker model is unavailable")
        if request.top_k > len(request.passages):
            raise RetrievalRequestError("reranking top_k exceeds passage count")
        identifiers = [passage.identifier for passage in request.passages]
        if len(set(identifiers)) != len(identifiers):
            raise RetrievalRequestError("passage identifiers must be unique")

        body = await self._post_json(
            self.reranker_client,
            "/rerank",
            {
                "query": request.query,
                "texts": [passage.text for passage in request.passages],
                "truncate": True,
                "raw_scores": False,
                "return_text": False,
            },
        )
        if not isinstance(body, list) or len(body) != len(request.passages):
            raise _protocol_error()

        seen: set[int] = set()
        ranked: list[tuple[int, float]] = []
        for item in body:
            if not isinstance(item, dict):
                raise _protocol_error()
            index = item.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(identifiers)
                or index in seen
            ):
                raise _protocol_error()
            seen.add(index)
            ranked.append((index, _finite_number(item.get("score"))))
        if seen != set(range(len(identifiers))):
            raise _protocol_error()
        ranked.sort(key=lambda result: result[1], reverse=True)
        data = [
            {"identifier": identifiers[index], "score": score}
            for index, score in ranked[: request.top_k]
        ]
        return {"model": self.settings.reranker_model, "data": data}
