# ADR-0004: Provider-neutral inference metrics

- Status: Accepted
- Date: 2026-08-04

## Context

Phase 1 must measure local generation consistently across the deterministic
fake, llama.cpp, vLLM, and future adapters. Time to first token, total latency,
token throughput, context utilization, failures, finish reasons, and caller
cancellation all depend on the full streaming lifecycle. Implementing those
measurements separately inside provider adapters would duplicate terminal-state
handling and allow provider payloads or PHI-bearing values to leak upward.

Prometheus labels are durable, indexed metadata. Any prompt, response, request
metadata, identifier, URL, or exception message in a label would create both an
information-disclosure risk and unbounded cardinality. The public web proxy must
also not expose the scrape surface.

## Decision

Wrap `ChatModelPort` implementations in an `InstrumentedChatModelAdapter`
decorator selected by `ModelAdapterRegistry`. The decorator owns lifecycle
measurement and emits only immutable, provider-neutral values through
`InferenceMetricsPort`. A monotonic clock is injectable for deterministic tests.

Normalize provider-reported `prompt_tokens` and `completion_tokens` into
`GenerationUsage` on the terminal `GenerationChunk`. OpenAI-compatible local
adapters request `stream_options.include_usage`; invalid usage fails closed as a
generic protocol error. If an engine completes without usage, request and
latency metrics remain valid while token-derived series are omitted. Argus does
not retokenize prompt or response text for metrics.

Record exactly one terminal outcome after stream success, adapter/protocol
failure, task cancellation, or caller close. Metrics failures are isolated from
inference and log only the metrics exception class at debug level. Disabled
deployments use a no-op sink.

The Prometheus adapter permits only:

- provider from `fake`, `llama_cpp`, or `vllm`;
- a configured model name matching the bounded settings contract;
- outcome from `success`, `failure`, or `cancelled`; and
- finish reason from `stop` or `length`.

Prompts, outputs, request metadata, document IDs, user IDs, trace IDs, exception
messages, endpoints, and free-form values are not accepted as labels or metric
attributes.

Expose the private registry at `/internal/metrics`, omit the route from OpenAPI,
and return 404 when metrics are disabled. The API remains reachable only on the
internal processing network, and nginx denies the entire `/api/internal/`
namespace. Prometheus/Grafana service provisioning remains Phase 6 work.

## Consequences

- Fake, llama.cpp, and vLLM follow one lifecycle and label policy.
- Cancellation and incomplete streams are visible without changing model
  behavior or exposing content.
- Token series depend on trusted local-engine usage data and may be absent for a
  request.
- Model names become a validated, bounded configuration value.
- The future Langfuse adapter can reuse the decorator pattern but requires a
  separate PHI-safe trace port and lifecycle decision before Phase 1 closes.
