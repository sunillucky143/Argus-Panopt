"""Deterministic inference adapter for tests and local contract development."""

from collections.abc import AsyncIterator

from app.domain.inference import (
    GenerationChunk,
    GenerationRequest,
    GenerationUsage,
    ModelCapabilities,
)


class FakeModelAdapter:
    """Return deterministic chunks without inspecting or logging prompt content."""

    def __init__(
        self,
        *,
        model_name: str = "argus-fake-model",
        max_context: int = 32_768,
        response: str = "Local inference adapter is ready.",
        ready: bool = True,
    ) -> None:
        self._model_name = model_name
        self._max_context = max_context
        self._response = response
        self._ready = ready

    async def health(self) -> bool:
        """Return deterministic readiness for health and integration tests."""

        return self._ready

    async def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Return a fresh asynchronous stream for each request."""

        del request

        async def stream() -> AsyncIterator[GenerationChunk]:
            words = self._response.split()
            for index, word in enumerate(words):
                suffix = "" if index == len(words) - 1 else " "
                yield GenerationChunk(text=f"{word}{suffix}", index=index)
            yield GenerationChunk(
                text="",
                index=len(words),
                finish_reason="stop",
                usage=GenerationUsage(input_tokens=0, output_tokens=len(words)),
            )

        return stream()

    async def capabilities(self) -> ModelCapabilities:
        """Describe the deterministic adapter."""

        return ModelCapabilities(
            provider="fake",
            model_name=self._model_name,
            max_context=self._max_context,
            vision=False,
            streaming=True,
            quantization="none",
            speculative_decoding=False,
            prefix_caching=False,
        )
