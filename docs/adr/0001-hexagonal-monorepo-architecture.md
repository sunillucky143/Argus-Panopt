# ADR-0001: Hexagonal monorepo architecture

- Status: Accepted
- Date: 2026-07-30

## Context

Argus Panopt handles potentially regulated documents while supporting multiple
local inference engines and hardware tiers. The pilot must remain testable
without a GPU or real model, and future engineers must be able to replace
infrastructure without rewriting business rules.

## Decision

Use a monorepo with a Python 3.12 FastAPI backend and strict TypeScript
React/Vite frontend. Organize backend behavior as a hexagonal architecture:

- `domain/` owns business entities and rules and imports no infrastructure.
- `ports/` defines provider-agnostic protocols.
- `adapters/` implements persistence, cache, identity, parsing, tracing, and
  local model access.
- `api/` and `workers/` are inbound adapters that compose use cases.
- construction occurs at application boundaries through dependency injection.

PostgreSQL with pgvector is the system of record; Redis provides disposable
caches, rate limits, and queue state. All AI functions are locally served behind
ports. Docker Compose is the single-node deployment boundary, with `cpu` and
`gpu` profiles sharing the same application interfaces.

Configuration is environment-only and validated on process startup. Network
egress is denied for processing services. Security and audit controls are
implemented with each feature, not deferred to a later hardening pass.

## Consequences

- Domain tests remain fast and do not require containers or model weights.
- Engine and model swaps are configuration changes behind contract-tested ports.
- Adapters add some structural overhead and require disciplined import
  direction.
- Compose is appropriate for the pilot but intentionally does not solve
  multi-node orchestration.
- The repository must maintain dependency direction tests and adapter contracts
  as the codebase grows.
