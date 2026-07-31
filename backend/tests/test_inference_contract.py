import asyncio

import pytest

from app.adapters.inference.fake import FakeModelAdapter
from app.adapters.inference.registry import (
    ModelAdapterRegistry,
    ModelAdapterUnavailableError,
    create_default_registry,
)
from app.core.config import Settings
from app.domain.inference import ChatMessage, GenerationChunk, GenerationRequest


def _request() -> GenerationRequest:
    return GenerationRequest(messages=(ChatMessage(role="user", content="status"),))


def test_fake_adapter_streams_deterministic_normalized_chunks() -> None:
    async def collect() -> list[GenerationChunk]:
        adapter = FakeModelAdapter(response="known answer")
        stream = await adapter.generate(_request())
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect())

    assert "".join(chunk.text for chunk in chunks) == "known answer"
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert chunks[-1].finish_reason == "stop"


def test_fake_adapter_reports_capabilities() -> None:
    capabilities = asyncio.run(
        FakeModelAdapter(model_name="test-model", max_context=4096).capabilities()
    )

    assert capabilities.provider == "fake"
    assert capabilities.model_name == "test-model"
    assert capabilities.max_context == 4096
    assert capabilities.streaming is True
    assert capabilities.vision is False


def test_fake_adapter_reports_configured_health() -> None:
    assert asyncio.run(FakeModelAdapter(ready=True).health()) is True
    assert asyncio.run(FakeModelAdapter(ready=False).health()) is False


def test_registry_reuses_configured_adapter() -> None:
    registry = create_default_registry()
    settings = Settings(
        environment="test",
        model_provider="fake",
        model_name="deterministic",
    )

    assert registry.resolve(settings) is registry.resolve(settings)


def test_registry_rejects_unregistered_provider_without_provider_details() -> None:
    registry = ModelAdapterRegistry()
    settings = Settings(environment="test", model_provider="llama_cpp")

    with pytest.raises(
        ModelAdapterUnavailableError,
        match=r"Configured local model provider is unavailable\.",
    ):
        registry.resolve(settings)
