"""Adapters for the deployment-local embedding and reranking service."""

import json
import math
from typing import Any

import httpx

from app.core.urls import validate_internal_service_endpoint
from app.domain.inference import EmbeddingKind, Passage, ScoredPassage, Vector

_MAX_ITEMS = 256
_MAX_TEXT_CHARACTERS = 32_768
_MAX_IDENTIFIER_CHARACTERS = 512
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class RetrievalAdapterError(RuntimeError):
    """Base error for content-safe retrieval adapter failures."""


class RetrievalProtocolError(RetrievalAdapterError):
    """Raised when the local service returns an invalid response."""


class InvalidRetrievalRequestError(RetrievalAdapterError):
    """Raised when a request exceeds the bounded local contract."""


def _protocol_error() -> RetrievalProtocolError:
    return RetrievalProtocolError("Local retrieval service returned an invalid response.")


def _validate_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRetrievalRequestError(f"{label} must be non-empty text.")
    if len(value) > _MAX_TEXT_CHARACTERS:
        raise InvalidRetrievalRequestError(f"{label} exceeds the character limit.")
    return value


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _protocol_error()
    number = float(value)
    if not math.isfinite(number):
        raise _protocol_error()
    return number


class _LocalRetrievalClient:
    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        transport: httpx.AsyncBaseTransport | None,
        timeout_seconds: float,
    ) -> None:
        stripped_name = model_name.strip()
        if not stripped_name or len(model_name) > _MAX_IDENTIFIER_CHARACTERS:
            raise ValueError("local retrieval model name is invalid")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("local retrieval timeout must be positive and finite")

        internal_endpoint = validate_internal_service_endpoint(
            endpoint,
            label="retrieval endpoint",
        )
        self.endpoint = f"{internal_endpoint}/"
        self.model_name = model_name
        self.transport = transport
        self.timeout = httpx.Timeout(timeout_seconds)

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        chunks: list[bytes] = []
        try:
            async with (
                httpx.AsyncClient(
                    base_url=self.endpoint,
                    transport=self.transport,
                    timeout=self.timeout,
                    follow_redirects=False,
                    trust_env=False,
                ) as client,
                client.stream("POST", path, json=payload) as response,
            ):
                response.raise_for_status()
                response_size = 0
                async for chunk in response.aiter_bytes():
                    response_size += len(chunk)
                    if response_size > _MAX_RESPONSE_BYTES:
                        raise _protocol_error()
                    chunks.append(chunk)
        except RetrievalProtocolError:
            raise
        except httpx.HTTPError as error:
            raise RetrievalAdapterError("Local retrieval service request failed.") from error

        try:
            body = json.loads(b"".join(chunks))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _protocol_error() from error
        if not isinstance(body, dict):
            raise _protocol_error()
        return body


class EmbeddingServiceAdapter:
    """Translate embedding requests to the deployment-local HTTP contract."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = _LocalRetrievalClient(
            endpoint=endpoint,
            model_name=model_name,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )

    async def embed(self, texts: list[str], kind: EmbeddingKind) -> list[Vector]:
        """Return ordered, finite vectors without exposing text in errors."""

        if kind not in {"query", "passage"}:
            raise InvalidRetrievalRequestError("embedding kind is unsupported")
        if not texts or len(texts) > _MAX_ITEMS:
            raise InvalidRetrievalRequestError("embedding batch size is invalid")
        validated_texts = [_validate_text(text, label="embedding input") for text in texts]

        body = await self._client.post(
            "v1/embeddings",
            {
                "model": self._client.model_name,
                "texts": validated_texts,
                "kind": kind,
            },
        )
        if body.get("model") != self._client.model_name:
            raise _protocol_error()
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise _protocol_error()

        vectors: dict[int, Vector] = {}
        dimension: int | None = None
        for item in data:
            if not isinstance(item, dict):
                raise _protocol_error()
            index = item.get("index")
            raw_vector = item.get("embedding")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(texts)
                or index in vectors
                or not isinstance(raw_vector, list)
                or not raw_vector
            ):
                raise _protocol_error()

            vector = tuple(_finite_number(value) for value in raw_vector)
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise _protocol_error()
            vectors[index] = vector

        if set(vectors) != set(range(len(texts))):
            raise _protocol_error()
        return [vectors[index] for index in range(len(texts))]


class RerankerServiceAdapter:
    """Translate reranking requests to the deployment-local HTTP contract."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = _LocalRetrievalClient(
            endpoint=endpoint,
            model_name=model_name,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )

    async def rerank(
        self,
        query: str,
        passages: list[Passage],
        top_k: int,
    ) -> list[ScoredPassage]:
        """Return bounded passages in descending finite-score order."""

        validated_query = _validate_text(query, label="reranking query")
        if not passages or len(passages) > _MAX_ITEMS:
            raise InvalidRetrievalRequestError("reranking passage count is invalid")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(passages):
            raise InvalidRetrievalRequestError("reranking top_k is invalid")

        passage_by_id: dict[str, Passage] = {}
        request_passages: list[dict[str, str]] = []
        for passage in passages:
            identifier = passage.identifier
            if (
                not isinstance(identifier, str)
                or not identifier
                or len(identifier) > _MAX_IDENTIFIER_CHARACTERS
                or identifier in passage_by_id
            ):
                raise InvalidRetrievalRequestError("passage identifiers must be unique and valid")
            text = _validate_text(passage.text, label="passage text")
            passage_by_id[identifier] = passage
            request_passages.append({"identifier": identifier, "text": text})

        body = await self._client.post(
            "v1/rerank",
            {
                "model": self._client.model_name,
                "query": validated_query,
                "passages": request_passages,
                "top_k": top_k,
            },
        )
        if body.get("model") != self._client.model_name:
            raise _protocol_error()
        data = body.get("data")
        if not isinstance(data, list) or len(data) != top_k:
            raise _protocol_error()

        seen: set[str] = set()
        results: list[ScoredPassage] = []
        for item in data:
            if not isinstance(item, dict):
                raise _protocol_error()
            result_identifier = item.get("identifier")
            if (
                not isinstance(result_identifier, str)
                or result_identifier not in passage_by_id
                or result_identifier in seen
            ):
                raise _protocol_error()
            seen.add(result_identifier)
            results.append(
                ScoredPassage(
                    passage=passage_by_id[result_identifier],
                    score=_finite_number(item.get("score")),
                )
            )

        return sorted(results, key=lambda result: result.score, reverse=True)
