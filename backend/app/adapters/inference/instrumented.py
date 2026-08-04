"""Provider-neutral instrumentation for streaming chat model adapters."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable

from app.domain.inference import (
    FinishReason,
    GenerationChunk,
    GenerationRequest,
    GenerationUsage,
    InferenceMetricDimensions,
    InferenceOutcome,
    InferenceRequestMetrics,
    ModelCapabilities,
)
from app.ports.inference import ChatModelPort, InferenceMetricsPort

logger = logging.getLogger(__name__)

Clock = Callable[[], float]


class InstrumentedChatModelAdapter:
    """Measure every chat provider through the same port-level decorator."""

    def __init__(
        self,
        model: ChatModelPort,
        metrics: InferenceMetricsPort,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self._model = model
        self._metrics = metrics
        self._clock = clock

    async def health(self) -> bool:
        """Delegate readiness without counting it as generation traffic."""

        return await self._model.health()

    async def capabilities(self) -> ModelCapabilities:
        """Preserve the wrapped provider-neutral capability contract."""

        return await self._model.capabilities()

    async def generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        """Return a stream that records completion, failure, and cancellation."""

        capabilities = await self._model.capabilities()
        dimensions = InferenceMetricDimensions(
            provider=capabilities.provider,
            model_name=capabilities.model_name,
        )
        started_at = self._clock()
        self._safe_started(dimensions)

        try:
            source = await self._model.generate(request)
        except asyncio.CancelledError:
            self._safe_finished(
                dimensions,
                self._terminal_metrics("cancelled", started_at, capabilities.max_context),
            )
            raise
        except Exception:
            self._safe_finished(
                dimensions,
                self._terminal_metrics("failure", started_at, capabilities.max_context),
            )
            raise

        async def stream() -> AsyncIterator[GenerationChunk]:
            first_token_at: float | None = None
            usage: GenerationUsage | None = None
            finish_reason: FinishReason | None = None
            terminal_recorded = False

            def finish(outcome: InferenceOutcome) -> None:
                nonlocal terminal_recorded
                if terminal_recorded:
                    return
                terminal_recorded = True
                self._safe_finished(
                    dimensions,
                    self._terminal_metrics(
                        outcome,
                        started_at,
                        capabilities.max_context,
                        first_token_at=first_token_at,
                        usage=usage,
                        finish_reason=finish_reason,
                    ),
                )

            try:
                async for chunk in source:
                    if chunk.text and first_token_at is None:
                        first_token_at = self._clock()
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.finish_reason is not None:
                        finish_reason = chunk.finish_reason
                    yield chunk
                finish("success" if finish_reason is not None else "failure")
            except (asyncio.CancelledError, GeneratorExit):
                finish("cancelled")
                raise
            except Exception:
                finish("failure")
                raise

        return stream()

    def _terminal_metrics(
        self,
        outcome: InferenceOutcome,
        started_at: float,
        context_ceiling: int,
        *,
        first_token_at: float | None = None,
        usage: GenerationUsage | None = None,
        finish_reason: FinishReason | None = None,
    ) -> InferenceRequestMetrics:
        finished_at = self._clock()
        total_latency = max(0.0, finished_at - started_at)
        time_to_first_token = (
            max(0.0, first_token_at - started_at) if first_token_at is not None else None
        )
        input_tokens = usage.input_tokens if usage is not None else None
        output_tokens = usage.output_tokens if usage is not None else None
        generation_seconds = (
            max(0.0, finished_at - first_token_at) if first_token_at is not None else 0.0
        )
        tokens_per_second = (
            output_tokens / generation_seconds
            if output_tokens is not None and generation_seconds > 0.0
            else None
        )
        context_utilization = (
            input_tokens / context_ceiling
            if input_tokens is not None and context_ceiling > 0
            else None
        )
        return InferenceRequestMetrics(
            outcome=outcome,
            total_latency_seconds=total_latency,
            time_to_first_token_seconds=time_to_first_token,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            context_utilization=context_utilization,
            finish_reason=finish_reason,
        )

    def _safe_started(self, dimensions: InferenceMetricDimensions) -> None:
        try:
            self._metrics.request_started(dimensions)
        except Exception as error:  # pragma: no cover - defensive adapter isolation
            logger.debug("Inference metrics emission failed: %s", type(error).__name__)

    def _safe_finished(
        self,
        dimensions: InferenceMetricDimensions,
        metrics: InferenceRequestMetrics,
    ) -> None:
        try:
            self._metrics.request_finished(dimensions, metrics)
        except Exception as error:  # pragma: no cover - defensive adapter isolation
            logger.debug("Inference metrics emission failed: %s", type(error).__name__)
