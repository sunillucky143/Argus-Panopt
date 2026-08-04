"""Prometheus and disabled adapters for inference metrics."""

import re

from prometheus_client import CollectorRegistry, Counter, Histogram

from app.domain.inference import InferenceMetricDimensions, InferenceRequestMetrics

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0)
TOKEN_BUCKETS = (1.0, 8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0, 32768.0, 131072.0)
RATE_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0)
UTILIZATION_BUCKETS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
ALLOWED_PROVIDERS = frozenset({"fake", "llama_cpp", "vllm"})
MODEL_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


class NoOpInferenceMetrics:
    """Discard metrics for explicitly disabled deployments and focused tests."""

    def request_started(self, dimensions: InferenceMetricDimensions) -> None:
        del dimensions

    def request_finished(
        self,
        dimensions: InferenceMetricDimensions,
        metrics: InferenceRequestMetrics,
    ) -> None:
        del dimensions, metrics


class PrometheusInferenceMetrics:
    """Emit content-free inference measurements into a private registry."""

    def __init__(self, registry: CollectorRegistry) -> None:
        labels = ("provider", "model")
        self._requests = Counter(
            "argus_inference_requests_total",
            "Generation requests accepted by the model port.",
            labels,
            registry=registry,
        )
        self._failures = Counter(
            "argus_inference_failures_total",
            "Generation requests ending in an adapter or protocol failure.",
            labels,
            registry=registry,
        )
        self._cancellations = Counter(
            "argus_inference_cancellations_total",
            "Generation streams cancelled or closed by the caller.",
            labels,
            registry=registry,
        )
        self._completions = Counter(
            "argus_inference_completions_total",
            "Successful generations by bounded finish reason.",
            (*labels, "finish_reason"),
            registry=registry,
        )
        self._latency = Histogram(
            "argus_inference_generation_latency_seconds",
            "Total model-port generation latency by bounded outcome.",
            (*labels, "outcome"),
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self._time_to_first_token = Histogram(
            "argus_inference_time_to_first_token_seconds",
            "Time from model-port request to the first non-empty token chunk.",
            labels,
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self._input_tokens = Histogram(
            "argus_inference_input_tokens",
            "Provider-reported input token count.",
            labels,
            buckets=TOKEN_BUCKETS,
            registry=registry,
        )
        self._output_tokens = Histogram(
            "argus_inference_output_tokens",
            "Provider-reported output token count.",
            labels,
            buckets=TOKEN_BUCKETS,
            registry=registry,
        )
        self._tokens_per_second = Histogram(
            "argus_inference_tokens_per_second",
            "Provider-reported output tokens divided by post-first-token duration.",
            labels,
            buckets=RATE_BUCKETS,
            registry=registry,
        )
        self._context_utilization = Histogram(
            "argus_inference_context_utilization_ratio",
            "Provider-reported input tokens divided by the configured context ceiling.",
            labels,
            buckets=UTILIZATION_BUCKETS,
            registry=registry,
        )

    def request_started(self, dimensions: InferenceMetricDimensions) -> None:
        self._requests.labels(*self._label_values(dimensions)).inc()

    def request_finished(
        self,
        dimensions: InferenceMetricDimensions,
        metrics: InferenceRequestMetrics,
    ) -> None:
        label_values = self._label_values(dimensions)
        self._latency.labels(*label_values, metrics.outcome).observe(metrics.total_latency_seconds)
        if metrics.outcome == "failure":
            self._failures.labels(*label_values).inc()
        elif metrics.outcome == "cancelled":
            self._cancellations.labels(*label_values).inc()
        elif metrics.finish_reason is not None:
            self._completions.labels(*label_values, metrics.finish_reason).inc()

        if metrics.time_to_first_token_seconds is not None:
            self._time_to_first_token.labels(*label_values).observe(
                metrics.time_to_first_token_seconds
            )
        if metrics.input_tokens is not None:
            self._input_tokens.labels(*label_values).observe(metrics.input_tokens)
        if metrics.output_tokens is not None:
            self._output_tokens.labels(*label_values).observe(metrics.output_tokens)
        if metrics.tokens_per_second is not None:
            self._tokens_per_second.labels(*label_values).observe(metrics.tokens_per_second)
        if metrics.context_utilization is not None:
            self._context_utilization.labels(*label_values).observe(metrics.context_utilization)

    @staticmethod
    def _label_values(dimensions: InferenceMetricDimensions) -> tuple[str, str]:
        if (
            dimensions.provider not in ALLOWED_PROVIDERS
            or MODEL_LABEL_PATTERN.fullmatch(dimensions.model_name) is None
        ):
            raise ValueError("Invalid inference metric dimensions.")
        return dimensions.provider, dimensions.model_name
