"""Ports for all deployment-local model access."""

from collections.abc import AsyncIterator
from typing import Protocol

from app.domain.inference import (
    EmbeddingKind,
    GenerationChunk,
    GenerationRequest,
    ImageRef,
    ModelCapabilities,
    Passage,
    ScoredPassage,
    Vector,
    VisionResult,
)


class ChatModelPort(Protocol):
    """Streaming text-generation boundary."""

    async def health(self) -> bool:
        """Return whether the configured local engine is ready to generate."""

    async def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Create a provider-neutral asynchronous result stream."""

    async def capabilities(self) -> ModelCapabilities:
        """Report features used for readiness and request planning."""


class EmbeddingPort(Protocol):
    """Text embedding boundary."""

    async def embed(self, texts: list[str], kind: EmbeddingKind) -> list[Vector]:
        """Embed query or passage text without exposing provider payloads."""


class RerankerPort(Protocol):
    """Passage reranking boundary."""

    async def rerank(
        self,
        query: str,
        passages: list[Passage],
        top_k: int,
    ) -> list[ScoredPassage]:
        """Return the most relevant passages in descending score order."""


class VisionPort(Protocol):
    """Deployment-local vision boundary."""

    async def describe(self, images: list[ImageRef], instruction: str) -> VisionResult:
        """Describe images without allowing an external network call."""
