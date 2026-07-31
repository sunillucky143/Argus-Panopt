"""Streaming adapters for deployment-local OpenAI-compatible model servers."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.domain.inference import GenerationChunk, GenerationRequest, ModelCapabilities


class ModelAdapterError(RuntimeError):
    """Base error for safe inference adapter failures."""


class ModelProtocolError(ModelAdapterError):
    """Raised when a local engine returns an invalid streaming response."""


class UnsupportedModelRequestError(ModelAdapterError):
    """Raised when a request needs adapter behavior not implemented yet."""


class OpenAICompatibleChatAdapter:
    """Translate provider-neutral requests to a local Chat Completions stream."""

    def __init__(
        self,
        *,
        endpoint: str,
        capabilities: ModelCapabilities,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._endpoint = f"{endpoint.rstrip('/')}/"
        self._capabilities = capabilities
        self._transport = transport
        self._timeout = httpx.Timeout(timeout_seconds)

    async def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Create a normalized stream without exposing prompt content in errors."""

        if request.images:
            raise UnsupportedModelRequestError(
                "Image generation requests require the dedicated local vision adapter."
            )

        payload = self._payload(request)

        async def stream() -> AsyncIterator[GenerationChunk]:
            try:
                async with (
                    httpx.AsyncClient(
                        base_url=self._endpoint,
                        transport=self._transport,
                        timeout=self._timeout,
                    ) as client,
                    client.stream(
                        "POST",
                        "chat/completions",
                        json=payload,
                    ) as response,
                ):
                    response.raise_for_status()
                    async for chunk in self._chunks(response):
                        yield chunk
            except ModelAdapterError:
                raise
            except httpx.HTTPError as error:
                raise ModelAdapterError("Local inference service request failed.") from error

        return stream()

    async def capabilities(self) -> ModelCapabilities:
        """Return the configured local engine capabilities."""

        return self._capabilities

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        messages = [
            {"role": message.role, "content": message.content} for message in request.messages
        ]
        if request.system is not None:
            messages.insert(0, {"role": "system", "content": request.system})

        payload: dict[str, Any] = {
            "model": self._capabilities.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        return payload

    async def _chunks(self, response: httpx.Response) -> AsyncIterator[GenerationChunk]:
        index = 0
        finished = False
        async for line in response.aiter_lines():
            if not line or line.startswith(":") or line.startswith("event:"):
                continue
            if not line.startswith("data:"):
                raise ModelProtocolError(
                    "Local inference service returned an invalid event stream."
                )

            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                if not finished:
                    yield GenerationChunk(text="", index=index, finish_reason="stop")
                return

            choice = self._first_choice(data)
            delta = choice.get("delta")
            if delta is not None and not isinstance(delta, dict):
                raise ModelProtocolError(
                    "Local inference service returned an invalid event stream."
                )

            content = delta.get("content") if delta else None
            if content is not None and not isinstance(content, str):
                raise ModelProtocolError(
                    "Local inference service returned an invalid event stream."
                )
            if content:
                yield GenerationChunk(text=content, index=index)
                index += 1

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if finish_reason not in {"stop", "length"}:
                    raise ModelProtocolError(
                        "Local inference service returned an unsupported finish reason."
                    )
                yield GenerationChunk(text="", index=index, finish_reason=finish_reason)
                finished = True

        if not finished:
            raise ModelProtocolError("Local inference service ended an incomplete event stream.")

    @staticmethod
    def _first_choice(data: str) -> dict[str, Any]:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise ModelProtocolError(
                "Local inference service returned an invalid event stream."
            ) from error
        if not isinstance(payload, dict):
            raise ModelProtocolError("Local inference service returned an invalid event stream.")

        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise ModelProtocolError("Local inference service returned an invalid event stream.")
        if not choices:
            return {}

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ModelProtocolError("Local inference service returned an invalid event stream.")
        return choice


class LlamaCppAdapter(OpenAICompatibleChatAdapter):
    """Local llama.cpp Chat Completions adapter."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        max_context: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            capabilities=ModelCapabilities(
                provider="llama_cpp",
                model_name=model_name,
                max_context=max_context,
                vision=False,
                streaming=True,
                quantization="GGUF Q4_K_M",
                speculative_decoding=False,
                prefix_caching=True,
            ),
            transport=transport,
        )


class VllmOpenAIAdapter(OpenAICompatibleChatAdapter):
    """Local vLLM Chat Completions adapter."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        max_context: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            capabilities=ModelCapabilities(
                provider="vllm",
                model_name=model_name,
                max_context=max_context,
                vision=False,
                streaming=True,
                quantization="AWQ",
                speculative_decoding=True,
                prefix_caching=True,
            ),
            transport=transport,
        )
