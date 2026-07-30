# ADR-0002: Provider-neutral local model adapters

- Status: Accepted
- Date: 2026-07-30

## Context

Argus Panopt must support llama.cpp, vLLM, deterministic tests, and future local engines without
letting provider payloads enter domain or API code. Streaming behavior and capability discovery
must remain consistent, and no implementation may call a third-party AI service.

## Decision

Define immutable provider-neutral generation values in the domain layer and asynchronous ports for
chat, embeddings, reranking, and vision. Resolve adapters through a configuration-driven registry
owned by the application process. Keep engine-specific protocol translation inside adapter modules.

The diagnostic generation endpoint emits normalized server-sent events and is disabled unless
explicitly enabled. It is never exposed when the application environment is production. Model
endpoints remain on the deployment's internal, egress-blocked network.
A deterministic fake adapter is the first contract implementation and is used by tests.

## Consequences

- Domain and API code cannot depend on vLLM, llama.cpp, or model-specific request bodies.
- Model changes are configuration-only after their adapter factory is registered.
- Contract tests can exercise streaming without model weights or network access.
- The registry must reject unknown providers safely and adapters must not log prompt content.
