import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from app.adapters.inference.instrumented import InstrumentedChatModelAdapter
from app.adapters.inference.metrics import PrometheusInferenceMetrics
from app.core.config import Settings
from app.domain.inference import (
    ChatMessage,
    GenerationChunk,
    GenerationRequest,
    GenerationUsage,
    InferenceMetricDimensions,
    InferenceRequestMetrics,
    ModelCapabilities,
)
from app.main import create_app


class RecordingMetrics:
    def __init__(self) -> None:
        self.started: list[InferenceMetricDimensions] = []
        self.finished: list[tuple[InferenceMetricDimensions, InferenceRequestMetrics]] = []

    def request_started(self, dimensions: InferenceMetricDimensions) -> None:
        self.started.append(dimensions)

    def request_finished(
        self,
        dimensions: InferenceMetricDimensions,
        metrics: InferenceRequestMetrics,
    ) -> None:
        self.finished.append((dimensions, metrics))


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class UsageModel:
    def __init__(self, *, response: str = "local answer") -> None:
        self._response = response

    async def health(self) -> bool:
        return True

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            provider="fake",
            model_name="bounded-model",
            max_context=1000,
            vision=False,
            streaming=True,
            quantization="none",
            speculative_decoding=False,
            prefix_caching=False,
        )

    async def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        del request

        async def stream() -> AsyncIterator[GenerationChunk]:
            yield GenerationChunk(text=self._response, index=0)
            yield GenerationChunk(
                text="",
                index=1,
                finish_reason="stop",
                usage=GenerationUsage(input_tokens=100, output_tokens=4),
            )

        return stream()


class FailingStreamModel(UsageModel):
    async def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        del request

        async def stream() -> AsyncIterator[GenerationChunk]:
            yield GenerationChunk(text=self._response, index=0)
            raise RuntimeError("private provider response")

        return stream()


class FailingCallModel(UsageModel):
    async def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        del request
        raise RuntimeError("private provider response")


def _request(content: str = "private prompt") -> GenerationRequest:
    return GenerationRequest(messages=(ChatMessage(role="user", content=content),))


def test_instrumented_model_records_deterministic_success_metrics() -> None:
    sink = RecordingMetrics()

    async def collect() -> list[GenerationChunk]:
        model = InstrumentedChatModelAdapter(
            UsageModel(),
            sink,
            clock=SequenceClock(10.0, 10.25, 11.25),
        )
        stream = await model.generate(_request())
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect())

    assert chunks[-1].usage == GenerationUsage(input_tokens=100, output_tokens=4)
    assert sink.started == [InferenceMetricDimensions(provider="fake", model_name="bounded-model")]
    assert len(sink.finished) == 1
    dimensions, metrics = sink.finished[0]
    assert dimensions == sink.started[0]
    assert metrics.outcome == "success"
    assert metrics.finish_reason == "stop"
    assert metrics.total_latency_seconds == 1.25
    assert metrics.time_to_first_token_seconds == 0.25
    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 4
    assert metrics.tokens_per_second == 4.0
    assert metrics.context_utilization == 0.1


def test_instrumented_model_records_failure_without_content() -> None:
    sink = RecordingMetrics()

    async def collect() -> None:
        model = InstrumentedChatModelAdapter(
            FailingStreamModel(response="private generated answer"),
            sink,
            clock=SequenceClock(0.0, 0.1, 0.5),
        )
        stream = await model.generate(_request("private patient prompt"))
        async for _ in stream:
            pass

    with pytest.raises(RuntimeError, match="private provider response"):
        asyncio.run(collect())

    rendered = repr(sink.started) + repr(sink.finished)
    assert "private patient prompt" not in rendered
    assert "private generated answer" not in rendered
    assert "private provider response" not in rendered
    assert sink.finished[0][1].outcome == "failure"


def test_instrumented_model_records_failure_before_stream_creation() -> None:
    sink = RecordingMetrics()

    async def start() -> None:
        model = InstrumentedChatModelAdapter(
            FailingCallModel(),
            sink,
            clock=SequenceClock(0.0, 0.5),
        )
        await model.generate(_request("private patient prompt"))

    with pytest.raises(RuntimeError, match="private provider response"):
        asyncio.run(start())

    assert len(sink.finished) == 1
    assert sink.finished[0][1].outcome == "failure"
    assert "private patient prompt" not in repr(sink.finished)


def test_instrumented_model_records_client_close_as_cancellation() -> None:
    sink = RecordingMetrics()

    async def close_after_first_chunk() -> None:
        model = InstrumentedChatModelAdapter(
            UsageModel(),
            sink,
            clock=SequenceClock(1.0, 1.1, 1.2),
        )
        stream = await model.generate(_request())
        await anext(stream)
        await cast(AsyncGenerator[GenerationChunk, None], stream).aclose()

    asyncio.run(close_after_first_chunk())

    assert len(sink.finished) == 1
    assert sink.finished[0][1].outcome == "cancelled"


def test_prometheus_adapter_exposes_only_bounded_content_free_labels() -> None:
    registry = CollectorRegistry(auto_describe=True)
    sink = PrometheusInferenceMetrics(registry)

    async def collect() -> None:
        model = InstrumentedChatModelAdapter(
            UsageModel(response="private generated answer"),
            sink,
            clock=SequenceClock(0.0, 0.25, 1.25),
        )
        stream = await model.generate(_request("private patient prompt"))
        async for _ in stream:
            pass

    asyncio.run(collect())
    exposition = generate_latest(registry).decode()

    assert 'provider="fake"' in exposition
    assert 'model="bounded-model"' in exposition
    assert 'finish_reason="stop"' in exposition
    assert 'outcome="success"' in exposition
    assert "private patient prompt" not in exposition
    assert "private generated answer" not in exposition


def test_prometheus_adapter_counts_failures_and_cancellations() -> None:
    registry = CollectorRegistry(auto_describe=True)
    sink = PrometheusInferenceMetrics(registry)

    async def exercise_terminal_paths() -> None:
        failing_model = InstrumentedChatModelAdapter(
            FailingStreamModel(),
            sink,
            clock=SequenceClock(0.0, 0.1, 0.2),
        )
        failing_stream = await failing_model.generate(_request())
        with pytest.raises(RuntimeError):
            async for _ in failing_stream:
                pass

        cancelled_model = InstrumentedChatModelAdapter(
            UsageModel(),
            sink,
            clock=SequenceClock(1.0, 1.1, 1.2),
        )
        cancelled_stream = await cancelled_model.generate(_request())
        await anext(cancelled_stream)
        await cast(AsyncGenerator[GenerationChunk, None], cancelled_stream).aclose()

    asyncio.run(exercise_terminal_paths())
    exposition = generate_latest(registry).decode()

    assert 'argus_inference_failures_total{model="bounded-model",provider="fake"} 1.0' in exposition
    assert (
        'argus_inference_cancellations_total{model="bounded-model",provider="fake"} 1.0'
        in exposition
    )


def test_prometheus_adapter_rejects_unbounded_dimensions_without_echoing_them() -> None:
    registry = CollectorRegistry(auto_describe=True)
    sink = PrometheusInferenceMetrics(registry)
    dimensions = InferenceMetricDimensions(
        provider="private-provider",
        model_name="private patient prompt",
    )

    with pytest.raises(ValueError, match="Invalid inference metric dimensions") as caught:
        sink.request_started(dimensions)

    exposition = generate_latest(registry).decode()
    assert "private-provider" not in str(caught.value)
    assert "private patient prompt" not in str(caught.value)
    assert "private-provider" not in exposition
    assert "private patient prompt" not in exposition


def test_metrics_endpoint_is_private_content_free_and_not_documented() -> None:
    settings = Settings(
        environment="test",
        debug_inference_enabled=True,
        model_provider="fake",
        model_name="metrics-model",
        model_context_ceiling=4096,
    )
    client = TestClient(create_app(settings))
    prompt = "private patient prompt"

    with client.stream(
        "POST",
        "/v1/debug/generate",
        json={"messages": [{"role": "user", "content": prompt}]},
    ) as response:
        list(response.iter_lines())

    metrics_response = client.get("/internal/metrics")
    openapi_response = client.get("/openapi.json")

    assert response.status_code == 200
    assert metrics_response.status_code == 200
    assert metrics_response.headers["cache-control"] == "no-store"
    assert metrics_response.headers["content-type"].startswith("text/plain")
    assert "argus_inference_requests_total" in metrics_response.text
    assert prompt not in metrics_response.text
    assert "/internal/metrics" not in openapi_response.json()["paths"]


def test_metrics_endpoint_is_hidden_when_disabled() -> None:
    client = TestClient(
        create_app(
            Settings(
                environment="test",
                debug_inference_enabled=True,
                inference_metrics_enabled=False,
                model_provider="fake",
            )
        )
    )

    generation_response = client.post(
        "/v1/debug/generate",
        json={"messages": [{"role": "user", "content": "private prompt"}]},
    )
    metrics_response = client.get("/internal/metrics")

    assert generation_response.status_code == 200
    assert metrics_response.status_code == 404
    assert metrics_response.json() == {"detail": "Not found."}


def test_web_proxy_explicitly_denies_internal_metrics() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    nginx_configuration = (repository_root / "frontend" / "nginx.conf").read_text()

    assert "location ^~ /api/internal/" in nginx_configuration
    assert "return 404;" in nginx_configuration


def test_metrics_sink_failure_does_not_break_inference_or_log_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RaisingMetrics:
        def request_started(self, dimensions: InferenceMetricDimensions) -> None:
            del dimensions
            raise RuntimeError("private metrics detail")

        def request_finished(
            self,
            dimensions: InferenceMetricDimensions,
            metrics: InferenceRequestMetrics,
        ) -> None:
            del dimensions, metrics
            raise RuntimeError("private metrics detail")

    async def collect() -> None:
        model = InstrumentedChatModelAdapter(
            UsageModel(),
            RaisingMetrics(),
            clock=SequenceClock(0.0, 0.1, 0.5),
        )
        stream = await model.generate(_request())
        async for _ in stream:
            pass

    with caplog.at_level(logging.DEBUG):
        asyncio.run(collect())

    assert "RuntimeError" in caplog.text
    assert "private metrics detail" not in caplog.text
